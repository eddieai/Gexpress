import cv2
import threading
from collections import deque

lock = threading.Lock()

class RealReadThread(threading.Thread):
    def __init__(self, input):
        super(RealReadThread).__init__()
        self._jobq = input
        # ip_camera_url = 'rtsp://admin:admin@192.168.1.64/'   # rtsp数据流
        # 创建一个窗口
        self.cap = cv2.VideoCapture(0)
        threading.Thread.__init__(self)

    def run(self):
        #        cv2.namedWindow('ip_camera') #, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        if not self.cap.isOpened():
            print('请检查IP地址还有端口号，或者查看IP摄像头是否开启，另外记得使用sudo权限运行脚本')
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            lock.acquire()
            if len(self._jobq) == 10:
                self._jobq.popleft()
            else:
                self._jobq.append(frame)
            lock.release()
            cv2.imshow('ip_camera', frame)
            if cv2.waitKey(1) == 27:
                # 退出程序
                break
        print("实时读取线程退出！！！！")
        cv2.destroyWindow('ip_camera')
        self._jobq.clear()  # 读取进程结束时清空队列
        self.cap.release()


class GetThread(threading.Thread):
    def __init__(self, input):
        super(GetThread).__init__()
        self._jobq = input
        threading.Thread.__init__(self)

    def run(self):
        #        cv2.namedWindow('get', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        flag = False
        while (True):
            if len(self._jobq) != 0:
                lock.acquire()
                im_new = self._jobq.pop()
                lock.release()
                cv2.imshow("get", im_new)
                cv2.waitKey(1000)
                flag = True
            elif flag == True and len(self._jobq) == 0:
                break
        print("间隔1s获取图像线程退出！！！！")


if __name__ == "__main__":
    q = deque([], 10)
    th1 = RealReadThread(q)
    th2 = GetThread(q)
    th1.start()
    th2.start()  # 开启两个线程

    th1.join()
    th2.join()
