# Constellation GUI

Run from the repository's root folder (or from any working directory):

```bash
pip install -r requirements.txt
pip install Pillow
python SNN_Authenticator/authenticator_gui.py
```

The interface uses the existing `saved_weights/2026-10-21.pth` model and `Auth_database` directory. Add **processed** database images with filenames `D<number>R<number>A<1-4>.jpg`, e.g. `D0001R1A1.jpg` through `D0001R1A4.jpg`. Each complete dot/replicate group is one search candidate. The database importer does not preprocess raw database photos; prepare these first using your existing dataset preprocessing workflow.

Choose **Processed** for four already-centered 224x224 grayscale query photos, or **Raw** for four original photos to be automatically thresholded, centered and cropped using the same algorithm as `Demo_Authenticator.py`. Select exactly four photos; they are sorted **alphabetically by filename** and assigned A1, A2, A3, A4, matching the original demo. Ensure filenames sort in the intended angle order (for example `query_A1.jpg` through `query_A4.jpg`).

Click **Run authentication**. The top five *dot/replicate* candidates are ranked by the **sum of four cosine similarities**. A candidate passes only if **each** angle score is at least `0.87`. Click a row to see that candidate's reference photo beside the query photo; use A1–A4 to switch angles. The GUI shows all four scores and the summed score. Similarities are not probabilities.

The GUI does not clear query or database folders. It uses a separate `gui_database_embeddings.pt` cache, automatically invalidated when reference files or model weights change. The original `Demo_Authenticator.py` remains untouched.

**Note:** Tkinter requires a local desktop Python installation (not a headless notebook/server). The application needs `Pillow` in addition to the root requirements. GPU is used automatically if PyTorch supports CUDA. Only load trusted PyTorch checkpoints.
