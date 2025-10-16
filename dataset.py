import os
import yaml

# Path to your dataset
dataset_path = r"E:\object dectection\Dataset"

# Define your class names here
class_names = ["class1", "class2", "class3"]  # 👈 change these to your classes

# Construct dataset structure
data_yaml = {
    "train": os.path.join(dataset_path, "images/train").replace("\\", "/"),
    "val": os.path.join(dataset_path, "images/val").replace("\\", "/"),
    "nc": len(class_names),
    "names": class_names
}

# Save YAML file
yaml_path = os.path.join(dataset_path, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"✅ data.yaml file created successfully at: {yaml_path}")
print("\nContents:")
print(yaml.dump(data_yaml, sort_keys=False))
