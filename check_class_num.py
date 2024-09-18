import os

folder = 'datasets256'

subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

for subfolder in subfolders:    # Traverse each subfolders
    folder_name = os.path.basename(subfolder)   # Get the name of the subfolders
    
    files = [f for f in os.scandir(subfolder) if f.is_file()]   # Retrieve all files in the subfolders
    
    image_count = sum(1 for f in files if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))   # Calculate the number of images

    if image_count == 0:        # If the number of images is 0, print out the names of the subfolders and delete them
        
        print(f"There are no images in the subfolders '{folder_name}', deleting them...")
        os.rmdir(subfolder)
        
        print(f"subfolders '{folder_name}' deleted")
    else:
        print(f"Number of images in subfolders: '{folder_name}' : {image_count}")