'''以1fps采样视频, 保存图片'''
from PIL import Image
import cv2
import json
import os
import glob
from tqdm import tqdm

gt_path = '/mnt/bn/duanbaishan-data/gengwenjia/dataset/qvhighlights/metadata/qvhighlights_val.jsonl'
parent_path = '/mnt/bn/duanbaishan-data/gengwenjia/dataset/qvhighlights/raw_data/'

video_dir = '/mnt/bn/experience0313/gengwenjia/timeblip_dataset/videos'
frame_dir = '/mnt/bn/experience0313/gengwenjia/timeblip_dataset/cooking_frames'
# 指定帧率（每秒采样的帧数）
frame_rate = 2  # 例如，每秒采样1帧

def save_video(video_path, output_folder, frame_rate):

    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 获取视频的帧率和总帧数
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算采样的帧间隔
    sample_interval = int(round(fps / frame_rate))

    # 设置起始帧
    current_frame = 0

    # 逐帧采样保存图片
    start_id = 0
    while current_frame < total_frames:
        # 读取当前帧
        ret, frame = video.read()

        # 如果达到采样间隔，则保存当前帧为图片
        if current_frame % sample_interval == 0:
            # 构造输出图片路径
            image_name = str(start_id)+".jpg"
            #output_path = os.path.join(output_folder, image_name)
            #os.makedirs(output_path, exist_ok=True)

            # 保存当前帧为图片
            #test_path = "/mnt/bn/duanbaishan-data/gengwenjia/dataset/qvhighlights/raw_data/test.jpg"
            cv2.imwrite(os.path.join(output_folder,image_name), frame)
            print("save image:", os.path.join(output_folder,image_name))
            start_id += 1
        
        # 更新当前帧
        current_frame += 1
    # 释放视频对象
    video.release()


'''首先可以使用json存储blip2推理的结果'''
if __name__ == '__main__':
    mp4_files = glob.glob(os.path.join(video_dir, '**', '*.avi'), recursive=True)
    for mp4_file in mp4_files:
        filename = os.path.splitext(os.path.basename(mp4_file))[0]
        frame_file = os.path.join(frame_dir, filename)
        os.makedirs(frame_file, exist_ok=True)
        save_video(mp4_file, frame_file, frame_rate)

    # with open(gt_path, 'r') as file:
    #     id = 0
        
    #     submissiion = []
    #     for line in tqdm(file):
    #         data = json.loads(line)
    #         qid = data["qid"]
    #         text_query = data["query"]
    #         vid_id = data["vid"]
    #         '''由video-id确认video-path'''
    #         vid_path = parent_path + 'videos/' + vid_id + '.mp4'
    #         out_path = parent_path + 'images/val/'+vid_id+'/'
    #         os.makedirs(out_path, exist_ok=True)
    #         save_video(vid_path, out_path, frame_rate)

