import os
import re
import decord
from decord import VideoReader, cpu

def change_file_name():
    directory_path = "D:\\AI\\mmaction2\\data\\Raw"
    index = 0
    i = 0
    j = 0

    for filename in os.listdir(directory_path):
        name = "Untitled video - Made with Clipchamp";
        if name in filename:
            id = int(re.search(r"\((\d+)\)", filename).group(1))
            new_filename = f"YT_J_{index}{os.path.splitext(filename)[1]}"
            index += 1
            old_file = os.path.join(directory_path, filename)
            new_file = os.path.join(directory_path, new_filename)

            print(f"Change {filename} to {new_filename}")

            os.rename(old_file, new_file)

def get_frame_decord(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = [frame.asnumpy() for frame in vr]
    return frames

def get_ave_frame():
    directory_path = "D:\\AI\\mmaction2\\data\\kinetics400_tiny\\train"

    frames = {}
    total_frame = 0
    index = 0
    for filename in os.listdir(directory_path):
        if os.path.splitext(filename)[1] == '.mp4':
            video_path = f'{directory_path}\\{filename}'
            frame = len(get_frame_decord(video_path))
            total_frame += frame
            index += 1
            frames.setdefault(filename, frame)

    print (frames)
    print (total_frame // index)

def get_file_name():
    directory_path = "D:\\AI\\mmaction2\\data\\malgo\\1"

    for filename in os.listdir(directory_path):
        if os.path.splitext(filename)[1] == '.mp4':
            print (filename)

if __name__ == "__main__":
    change_file_name()
    # get_ave_frame()
    # get_file_name()