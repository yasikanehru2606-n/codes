import os
from PIL import Image, ImageSequence

input_dataset = "filtered_data"  # source folder containing .webp/.gif subdirectories
output_dataset = "captured_data"  # destination folder for png frames (or jpeg if needed)

os.makedirs(output_dataset, exist_ok=True)

total_images = 0

for class_name in os.listdir(input_dataset):

    class_path = os.path.join(input_dataset, class_name)

    if not os.path.isdir(class_path):
        continue

    print("Processing:", class_name)

    output_class = os.path.join(output_dataset, class_name)
    os.makedirs(output_class, exist_ok=True)

    for file in os.listdir(class_path):

        if not (file.lower().endswith(".webp") or file.lower().endswith(".gif")):
            continue

        file_path = os.path.join(class_path, file)

        try:
            img = Image.open(file_path)

            frame_id = 0

            for frame in ImageSequence.Iterator(img):

                frame = frame.convert("RGB")

                save_path = os.path.join(
                    output_class,
                    f"{os.path.splitext(file)[0]}_{frame_id}.png"
                )

                frame.save(save_path)

                frame_id += 1
                total_images += 1

            print(f"Converted {file} → {frame_id} frames")

        except Exception as e:
            print("Error:", file, e)

print("\nTotal images created:", total_images)
print("Output dataset:", output_dataset)