import os
import cv2
import numpy as np


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

# if __name__=='__main__':
#     vidoe_dir = os.listdir('dataset/Masks')
#     # print(vidoe_dir)
#     for vid in vidoe_dir:
#         create_blank(os.path.join('dataset/Masks', vid))



        # for 
        # check_frames(video_root= os.path.join(video_root, vid), mask_root=os.path.join(mask_root, vid))


    # print(video_names)
    # print(mask_names)

    # for vid in mask_names:
    #     # print(vid)
    #     if vid not in video_names:
    #         print(vid)

# if __name__=='__main__':                                                                                                                                                                                           
#     vidoe_dir = os.listdir('TestWithNewAnnotations22')                                                                                                                                                             
#     # print(vidoe_dir)                                                                                                                                                                                             
#     for vid in vidoe_dir:                                                                                                                                                                                          
#         create_blank(os.path.join('TestWithNewAnnotations22', vid))                                                                                                                                                
                                                                         


if __name__== '__main__':
    # base_directory = 'dataset/mask/VID26'
    # create_log(base_directory=base_directory)

    # check_video_names(video_root='dataset/videos_batched', mask_root='dataset/Masks')
    check_video(video_root='dataset/videos_batched', mask_root='dataset/Masks')
