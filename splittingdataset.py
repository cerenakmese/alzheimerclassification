import os
import random
import shutil
import time

start_time=time.time()
print("Running")
dataset_path="/alzheimer_dataset"
main_dataset=os.path.join(dataset_path,"AugmentedAlzheimerDataset")
train_path=os.path.join(dataset_path,"train")
test_path=os.path.join(dataset_path,"test")

os.makedirs(train_path,exist_ok=True)
os.makedirs(test_path,exist_ok=True)
classes=os.listdir(main_dataset)

for class_name in classes:
    class_path=os.path.join(main_dataset,class_name)
    train_class_path=os.path.join(train_path,class_name)
    test_class_path=os.path.join(test_path,class_name)

    os.makedirs(train_class_path,exist_ok=True)
    os.makedirs(test_class_path,exist_ok=True)

    files=os.listdir(class_path)
    random.shuffle(files)

    test_ratio=0.1
    test_count=int(len(files)*test_ratio)
    test_files=files[:test_count]
    train_files=files[test_count:]

    for file in test_files:
        src=os.path.join(class_path,file)
        dst=os.path.join(test_class_path,file)
        shutil.copy(src,dst)

    for file in train_files:
        src=os.path.join(class_path,file)
        dst=os.path.join(train_class_path,file)
        shutil.copy(src,dst)

print("Finished.")
end_time = time.time()
elapsed_time=end_time-start_time
print(f"Elapsed time:{elapsed_time}")








