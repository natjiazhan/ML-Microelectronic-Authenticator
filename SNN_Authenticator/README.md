SNN Authenticator is a Python script for training SNN models and authenticating UP dot samples against a known database.


1 Training Mode

Use this mode to train new weights using the images in the Training_database folder. 
Images in this folder should already be processed (centered, grayscale, and put in 224x224 format).
By default, there is a placeholder in the Training_database folder. 
The full set of database images can be found here: https://drive.google.com/drive/folders/1lThHwgKsJLA64oFtv6bZgkKMqZPrgUQL
After training is complete, the weights will be saved in the saved_weights folder.

*Important Notes*

Change "select_weight" to select what weights you want to use for authentication.
To manually set the threshold, set "use_set_threshold" to True, and change "set_threshold" to the desired threshold value (typically 0.8-0.95). 

2 One to One Auth Mode

This mode is for getting the similarity of two images. If the similarity is above the threshold it is considered a match.
Put exactly two images in the one_to_one_auth folder before running the script.

3 Database Search Mode

This mode takes a set of 4 images in the Query_processed folder and compares it to other sets in the Auth_database folder to identify matches. 
The top 5 strongest matches and their similarity scores are shown. 
If all 4 images in the Query match with another set in Auth_database with similarity above the threshold, it is considered a database match.
Sample images in the Samples folder are already processed, and can be copied into the Query_processed folder to test the Autheticator.
For raw images, put them into the Query_raw folder, then run Query_img_processing.py before running the SNN_Authenticator.py.

4 Test Authenticator Mode

This mode tests every combination of the images in the Samples folder with the images in the Auth_database folder to measure authenticator performance.

Author: Nam Nguyen
