"""Tkinter front end for Demo_Authenticator's four-angle database search."""
from __future__ import annotations

import hashlib
import re
import shutil
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'Auth_database'
WEIGHTS = ROOT / 'saved_weights' / '2026-10-21.pth'
CACHE = ROOT / 'gui_database_embeddings.pt'
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
PATTERN = re.compile(r'D(\d+)R(\d+)A(\d+)', re.I)
THRESHOLD = 0.87


class CNN(nn.Module):
    """Exact embedding network from Demo_Authenticator.py."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x).view(x.size(0), -1)
        return F.normalize(self.fc(x), p=2, dim=1)


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Cannot read image: {path}')
    return img


def preprocess_raw(paths):
    """Same combine/centroid/240px crop/224px resize as the demo."""
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read image: {path}')
        images.append(image)
    combined = None
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.threshold(gray, 40, 255, cv2.THRESH_TOZERO)[1]
        combined = filtered if combined is None else cv2.add(combined, filtered)
    moments = cv2.moments(combined)
    if moments['m00'] == 0:
        raise ValueError('Unable to find dot center. Check illumination and image content.')
    center = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
    result = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.threshold(gray, 20, 255, cv2.THRESH_TOZERO)[1]
        crop = cv2.getRectSubPix(filtered, (240, 240), center)
        result.append(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA))
    return result


def database_sets():
    grouped = {}
    for path in sorted(DB.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXT:
            continue
        match = PATTERN.fullmatch(path.stem)
        if not match:
            raise ValueError(f'Invalid database filename: {path.name}. Expected D0001R1A1.jpg')
        dot, rep, angle = map(int, match.groups())
        if angle in (1, 2, 3, 4):
            grouped.setdefault((dot, rep), {})[angle] = path
    return {key: value for key, value in grouped.items() if len(value) == 4}


def search(query_images):
    start = time.perf_counter()
    if not WEIGHTS.exists():
        raise FileNotFoundError(f'Model checkpoint missing: {WEIGHTS}')
    groups = database_sets()
    if not groups:
        raise ValueError('No complete four-angle database sets found. Add D<number>R<number>A1..A4 images.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Checkpoint comes with this repository. Only load checkpoints from trusted sources.
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model = CNN().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    mean, std = float(checkpoint['mean']), float(checkpoint['std'])
    if std == 0:
        raise ValueError('Invalid checkpoint normalization: std=0')

    def embed(images):
        array = np.stack([(img.astype(np.float32) - mean) / std for img in images])
        with torch.no_grad():
            return model(torch.from_numpy(array).unsqueeze(1).to(device)).cpu()

    query_embeddings = embed(query_images)
    files = sorted((str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
                   for angles in groups.values() for path in angles.values())
    stat = WEIGHTS.stat()
    signature = hashlib.sha256(repr((str(WEIGHTS.resolve()), stat.st_mtime_ns, stat.st_size,
                                     mean, std, files)).encode()).hexdigest()
    embeddings = None
    if CACHE.exists():
        try:
            cache = torch.load(CACHE, map_location='cpu', weights_only=True)
            if cache.get('signature') == signature:
                embeddings = cache['embeddings']
        except (OSError, RuntimeError, KeyError, ValueError, EOFError):
            pass
    keys = list(groups)
    if embeddings is None:
        flattened = [(key, angle, groups[key][angle]) for key in keys for angle in (1, 2, 3, 4)]
        vectors = []
        for i in range(0, len(flattened), 32):
            vectors.append(embed([load_gray(item[2]) for item in flattened[i:i + 32]]))
        embeddings = torch.cat(vectors, dim=0)
        torch.save({'signature': signature, 'embeddings': embeddings}, CACHE)
    if embeddings.shape[0] != 4 * len(keys):
        raise ValueError('Database cache has an unexpected shape. Delete gui_database_embeddings.pt and retry.')
    sims = (embeddings.reshape(len(keys), 4, 128) * query_embeddings.unsqueeze(0)).sum(dim=2)
    results = []
    for i, key in enumerate(keys):
        scores = [float(v) for v in sims[i]]
        results.append({'key': key, 'scores': scores, 'total': sum(scores),
                        'pass': all(s >= THRESHOLD for s in scores), 'paths': groups[key]})
    results.sort(key=lambda item: item['total'], reverse=True)
    return results[:5], sum(r['pass'] for r in results), len(results), time.perf_counter() - start, str(device)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Constellation | Microelectronic Authenticator')
        self.geometry('1120x820')
        self.minsize(900, 700)
        self.configure(bg='#101827')
        self.query_paths = []
        self.query_images = []
        self.results = []
        self.selected = 0
        self.angle = tk.IntVar(value=1)
        self.mode = tk.StringVar(value='processed')
        self.status = tk.StringVar(value='Select four query images to begin.')
        self.db_status = tk.StringVar()
        self.photos = []
        self._style()
        self._build()
        self.refresh_db()

    def _style(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background='#101827')
        style.configure('TLabel', background='#101827', foreground='#e6edf7', font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI Semibold', 22), foreground='#f8fafc')
        style.configure('Muted.TLabel', foreground='#a6b3c7')
        style.configure('TButton', font=('Segoe UI Semibold', 10), padding=(14, 10))
        style.configure('Accent.TButton', background='#4f8cff', foreground='#ffffff')
        style.map('Accent.TButton', background=[('active', '#3974e4'), ('disabled', '#41536f')])
        style.configure('TRadiobutton', background='#101827', foreground='#e6edf7')
        style.configure('Treeview', background='#1a273b', foreground='#e6edf7', fieldbackground='#1a273b', rowheight=30, font=('Segoe UI', 10))
        style.configure('Treeview.Heading', background='#263954', foreground='#f8fafc', font=('Segoe UI Semibold', 10))
        style.map('Treeview', background=[('selected', '#365e9a')])

    def _build(self):
        outer = ttk.Frame(self, padding=22)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='CONSTELLATION', style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer, text='UP dot authentication  •  Siamese CNN  •  Four-angle database search', style='Muted.TLabel').pack(anchor='w', pady=(0, 17))
        bar = ttk.Frame(outer)
        bar.pack(fill='x', pady=(0, 9))
        ttk.Button(bar, text='＋  Add database images', command=self.add_database).pack(side='left', padx=(0, 9))
        ttk.Button(bar, text='▣  Select 4 query images', command=self.select_query).pack(side='left', padx=(0, 9))
        self.run_button = ttk.Button(bar, text='▶  Run authentication', style='Accent.TButton', command=self.run_search)
        self.run_button.pack(side='left')
        ttk.Label(bar, textvariable=self.db_status, style='Muted.TLabel').pack(side='right')
        modebar = ttk.Frame(outer)
        modebar.pack(fill='x', pady=(0, 12))
        ttk.Label(modebar, text='Query format:').pack(side='left', padx=(0, 10))
        ttk.Radiobutton(modebar, text='Already processed (224 × 224)', variable=self.mode, value='processed', command=self.reset_query).pack(side='left', padx=(0, 15))
        ttk.Radiobutton(modebar, text='Raw (auto preprocess)', variable=self.mode, value='raw', command=self.reset_query).pack(side='left')
        ttk.Label(outer, textvariable=self.status, style='Muted.TLabel', wraplength=1000).pack(anchor='w', pady=(0, 14))
        ttk.Label(outer, text='TOP DATABASE MATCHES').pack(anchor='w', pady=(0, 6))
        columns = ('rank', 'candidate', 'a1', 'a2', 'a3', 'a4', 'total', 'result')
        self.tree = ttk.Treeview(outer, columns=columns, show='headings', height=5, selectmode='browse')
        headings = ('#', 'Candidate', 'Angle 1', 'Angle 2', 'Angle 3', 'Angle 4', 'Total / 4', 'Auth')
        for col, label in zip(columns, headings):
            self.tree.heading(col, text=label)
            self.tree.column(col, width=100 if col not in ('rank', 'candidate', 'result') else (42 if col == 'rank' else 120), anchor='center', stretch=True)
        self.tree.pack(fill='x', pady=(0, 17))
        self.tree.bind('<<TreeviewSelect>>', self.choose_match)
        previewbar = ttk.Frame(outer)
        previewbar.pack(fill='x', pady=(0, 8))
        ttk.Label(previewbar, text='IMAGE COMPARISON').pack(side='left')
        ttk.Label(previewbar, text='Illumination angle:').pack(side='left', padx=(30, 8))
        for angle in (1, 2, 3, 4):
            ttk.Radiobutton(previewbar, text=f'A{angle}', variable=self.angle, value=angle, command=self.update_preview).pack(side='left', padx=5)
        self.preview = ttk.Frame(outer)
        self.preview.pack(fill='both', expand=True)
        self.query_panel = self._panel(self.preview, 'QUERY IMAGE', 0)
        self.match_panel = self._panel(self.preview, 'MATCHED DATABASE IMAGE', 1)
        ttk.Label(outer, text='Scores are cosine similarities (not calibrated probabilities). Authentication passes only when all four angles ≥ 0.87.', style='Muted.TLabel').pack(anchor='w', pady=(12, 0))

    def _panel(self, parent, title, column):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, sticky='nsew', padx=(0, 10) if column == 0 else (10, 0))
        parent.columnconfigure(column, weight=1)
        parent.rowconfigure(0, weight=1)
        ttk.Label(frame, text=title).pack(anchor='w', pady=(0, 8))
        label = tk.Label(frame, text='No image selected', bg='#1a273b', fg='#a6b3c7', font=('Segoe UI', 11), height=15)
        label.pack(fill='both', expand=True)
        return label

    def refresh_db(self):
        try:
            groups = database_sets()
            self.db_status.set(f'{len(groups)} complete reference sets')
        except Exception as exc:
            self.db_status.set('Database needs attention')
            messagebox.showerror('Database error', str(exc))

    def add_database(self):
        paths = filedialog.askopenfilenames(title='Add processed database images (D1R1A1.jpg, etc.)', filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')])
        if not paths:
            return
        invalid = [Path(p).name for p in paths if not PATTERN.fullmatch(Path(p).stem)]
        if invalid:
            messagebox.showerror('Invalid filenames', 'Database images must follow D<number>R<number>A<number>, e.g. D0001R1A1.jpg.\n\n' + ', '.join(invalid[:8]))
            return
        DB.mkdir(exist_ok=True)
        collisions = [Path(p).name for p in paths if (DB / Path(p).name).exists()]
        if collisions and not messagebox.askyesno('Replace files?', f'{len(collisions)} files already exist. Replace them?'):
            return
        try:
            for path in paths:
                shutil.copy2(path, DB / Path(path).name)
        except OSError as exc:
            messagebox.showerror('Import error', str(exc))
            return
        self.refresh_db()
        self.status.set(f'Imported {len(paths)} database images. Only complete A1–A4 sets participate in search.')

    def reset_query(self):
        self.query_paths = []
        self.query_images = []
        self.results = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.update_preview()
        self.status.set('Select exactly four query images, ordered A1 through A4 by filename.')

    def select_query(self):
        paths = filedialog.askopenfilenames(title='Select four query images (A1, A2, A3, A4)', filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')])
        if not paths:
            return
        if len(paths) != 4:
            messagebox.showerror('Four images required', f'Select exactly 4 images. You selected {len(paths)}.')
            return
        paths = sorted((Path(p) for p in paths), key=lambda p: p.name.lower())
        try:
            if self.mode.get() == 'raw':
                images = preprocess_raw(paths)
            else:
                images = [load_gray(path) for path in paths]
                if any(img.shape != (224, 224) for img in images):
                    raise ValueError('Processed query images must be 224 × 224. Select Raw mode for unprocessed images.')
        except Exception as exc:
            messagebox.showerror('Query error', str(exc))
            return
        self.reset_query()
        self.query_paths, self.query_images = paths, images
        self.status.set('Query ready: ' + '  |  '.join(f'A{i}: {p.name}' for i, p in enumerate(paths, 1)))
        self.update_preview()

    def run_search(self):
        if len(self.query_images) != 4:
            messagebox.showwarning('Query required', 'Select four query images first.')
            return
        self.run_button.configure(state='disabled')
        self.status.set('Running four-angle database search…')
        threading.Thread(target=self._worker, args=(list(self.query_images),), daemon=True).start()

    def _worker(self, images):
        try:
            outcome = search(images)
            self.after(0, lambda: self._complete(outcome))
        except Exception as exc:
            error = str(exc)
            self.after(0, lambda: self._failed(error))

    def _failed(self, error):
        self.run_button.configure(state='normal')
        self.status.set('Search failed. Please review the error.')
        messagebox.showerror('Authentication error', error)

    def _complete(self, outcome):
        self.run_button.configure(state='normal')
        self.results, passing, count, elapsed, device = outcome
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, result in enumerate(self.results):
            dot, rep = result['key']
            scores = result['scores']
            self.tree.insert('', 'end', iid=str(i), values=(i + 1, f'D{dot:04d}R{rep}',
                *(f'{v:.4f}' for v in scores), f'{result["total"]:.4f}', 'PASS' if result['pass'] else 'FAIL'))
        if self.results:
            self.tree.selection_set('0')
            self.tree.focus('0')
            self.selected = 0
        self.status.set(f'Searched {count} complete sets in {elapsed:.2f}s on {device.upper()}  •  {passing} candidate(s) passed all four angles.')
        self.update_preview()

    def choose_match(self, _event):
        selection = self.tree.selection()
        if selection:
            self.selected = int(selection[0])
            self.update_preview()

    def _show_image(self, label, image):
        picture = Image.fromarray(image).convert('RGB')
        picture.thumbnail((370, 340), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(picture)
        label.configure(image=photo, text='')
        label.image = photo

    def update_preview(self):
        angle = self.angle.get()
        if self.query_images:
            self._show_image(self.query_panel, self.query_images[angle - 1])
        else:
            self.query_panel.configure(image='', text='No query image selected')
        if self.results:
            try:
                image = load_gray(self.results[self.selected]['paths'][angle])
                self._show_image(self.match_panel, image)
            except Exception as exc:
                self.match_panel.configure(image='', text=f'Cannot load match: {exc}')
        else:
            self.match_panel.configure(image='', text='Run authentication to see a match')


if __name__ == '__main__':
    App().mainloop()
