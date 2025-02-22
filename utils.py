import os
import cv2
import numpy as np
import shutil

def create_log(base_directory):
    part_directories = [os.path.join(base_directory, dir) for dir in os.listdir(base_directory)]
    part_directories = sorted(part_directories)

    for dir in part_directories:
        frames = sorted(os.listdir(dir))
        first_frame_of_part = frames[0]
        count = 0
        for f in frames:
            count += 1
        print(f'the name of the dir: {dir}')
        print(f'the first frame was: {first_frame_of_part}')
        print(f'the count of that part of the dir was: {count}')
        print('------------------------------------------')                                                                                                                                                             
                                                                                                                                                                                                                   
def check_frames(video_root, mask_root):
    if not os.path.exists(mask_root):
        print(f'this path not found {mask_root}')
    else:
        video_names = [os.path.splitext(name)[0] for name in os.listdir(video_root)]
        mask_names = [os.path.splitext(name)[0] for name in os.listdir(mask_root)]

        missing_files = sorted(list(set(video_names) - set(mask_names)))

        print(f' for the path {mask_root}: {missing_files}')
        print('---------------------------------------------------------------------------------------')

def check_video(video_root , mask_root):
    for vid in sorted(os.listdir(video_root)):
        vid_path = os.path.join(video_root, vid)
        mask_vid_path = os.path.join(mask_root, vid)

        if os.path.exists(mask_vid_path):
            for part in os.listdir(vid_path):
                check_frames(video_root=os.path.join(vid_path, part), mask_root=os.path.join(mask_vid_path, part))
        else:
            print(f'path not found for {mask_vid_path}')

def generate_frame_name(frame_number):
    return f"{frame_number:06d}.png"

def create_blank(mask_dir):
    # mask_dir = 'VID04'
    print(mask_dir)
    parts = sorted(
        [part for part in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, part))],
        key=lambda x: int(x.replace('part', ''))
    )

    for part in parts:
        start_idx = (int(part[-1]) * 600) - 600
        part_directory = os.path.join(mask_dir, part)

        frames = [filename for filename in os.listdir(part_directory) if filename.endswith('.png')]
        frames.sort()

        first_frame_number = int(frames[0].split('.')[0]) 

        if first_frame_number > start_idx:
            print(f'the first frame num:{first_frame_number} and the path dir is {part_directory}')
            frame_shape = cv2.imread(os.path.join(part_directory, generate_frame_name(first_frame_number))).shape
            print(frame_shape)
            for i in range(start_idx, first_frame_number):
                frame_path  = os.path.join(part_directory, generate_frame_name(i))

                blank_frame = np.zeros(frame_shape, dtype=np.uint8)
                cv2.imwrite(frame_path, blank_frame)


# Removes the parts(no longer part1, part2 etc BS)
def merge_part_images(video_dir="video", valid_extensions={".png", ".jpg"}):
    """
    Moves all images from subdirectories inside `video_dir` to the main directory.
    Removes empty subdirectories after moving the images.
    
    Args:
        video_dir (str): The root directory containing subdirectories with images.
        valid_extensions (set): A set of valid image extensions (e.g., {".png", ".jpg"}).
    """
    # Get all subdirectories inside 'video'
    print(video_dir)
    parts = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]

    # Move all images from subdirectories to the video directory
    for part in parts:
        part_path = os.path.join(video_dir, part)
        for img in sorted(os.listdir(part_path)):  # Sort to maintain order
            img_path = os.path.join(part_path, img)
            if os.path.isfile(img_path) and os.path.splitext(img)[1].lower() in valid_extensions:
                shutil.move(img_path, os.path.join(video_dir, img))

    # Remove empty subdirectories
    for part in parts:
        part_path = os.path.join(video_dir, part)
        if os.path.exists(part_path) and not os.listdir(part_path):
            os.rmdir(part_path)

if __name__ == '__main__':
    videos = 'dataset/videos_batched'
    mask = 'dataset/Masks'

    for dir in os.listdir(videos):
        merge_part_images(video_dir=os.path.join(videos, dir))
        merge_part_images(video_dir=os.path.join(mask, dir))
        