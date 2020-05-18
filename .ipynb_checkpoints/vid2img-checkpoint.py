import cv2 as cv2
import os

video_path = '/home/jupyter/Drive_2/UCF_Dataset/Abnormal_Test/'
save_path = '/home/jupyter/Drive_2/Event_Detection/ucf_frames/test/abnormal/'

#video_path = '/home/jupyter/Accidents_Dataset/Accident/'
#save_path = '/home/jupyter/Drive_2//Event_Detection/accidents_dataset/Accident/'

action_list = os.listdir(video_path)
print("List of Directories: ", action_list)

target_classes = set(['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism' ])
#target_classes = set(['Normal'])
for action in action_list:
    print('Action: ', action)
    if action in target_classes:
        #if not os.path.exists(save_path+action):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            #os.mkdir(save_path+action)
        video_list = os.listdir(video_path+action)
        print ("Videos List: ", video_list)
        vid_count = 0
        for video in video_list:
            #if vid_count == 5:
            #    break
            print('Video: ', video)
            prefix = video.split('.')[0]
            #if not os.path.exists(save_path+ '/' + action + '/' +prefix):
            if not os.path.exists(save_path + '/' + prefix):
                #os.mkdir(save_path+ '/' + action+'/'+prefix)
                os.mkdir(save_path  + prefix)
            save_name = save_path + prefix + '/' + action
            #save_name = save_path + action + '/'
            video_name = video_path+action+'/'+ video
            cap = cv2.VideoCapture(video_name)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    cv_path = save_name+ '_' + str(10000+frame_count+(vid_count*10000))+'.jpg'
                    cv2.imwrite(cv_path, frame)
                    #print(save_name+str(10000+frame_count+(vid_count*10000))+'.jpg')
                    frame_count += 1
            vid_count += 1