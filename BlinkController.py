import threading
import time
import random
import math

lock = threading.Lock()

class BlinkController(threading.Thread):
    """
    Threaded realistic blink controller for a PyLiveLinkFace instance.
    - py_face: PyLiveLinkFace instance
    - send_callback: function(py_face) -> should transmit py_face.encode() to Unreal (optional)
    - stop_event: threading.Event used to stop the thread
    - intensity: max blink value (0.0..1.0)
    - mean_interval: average seconds between blinks
    """
    def __init__(self, set_blink, send_callback=None, stop_event=None, mean_interval=4.0):
        super().__init__(daemon=True)
        self.send_callback = send_callback
        self.set_blink = set_blink
        self.stop_event = stop_event or threading.Event()
        self.defaultintensity = 1.0
        self.mean_interval = float(mean_interval)
        # blendshape enum names used here - adapt if your enum uses different names
        

    def ease_cos(self, t):
        # cosine ease in/out: 0..1 -> 0..1 smooth
        return 0.5 - 0.5 * math.cos(math.pi * t)

    def do_blink(self):
        global left_int, right_int
        # randomized timings (seconds)
        close_time = random.uniform(0.06, 0.14)   # time to close
        closed_hold = random.uniform(0.02, 0.12)  # eyelid hold closed
        open_time = random.uniform(0.06, 0.14)    # time to open

        # small asynchrony between eyes
        asymmetry = random.uniform(-0.03, 0.03)
        step = 0.033
        # closing
        t = 0.0
        while t < close_time:
            if self.stop_event.is_set():
                return
            p = self.ease_cos(min(1.0, t / close_time))
            self.set_blink( self.defaultintensity * p,
                            self.defaultintensity * max(0.0, min(1.0, p + asymmetry)))
            if self.send_callback:
                self.send_callback()
            time.sleep(step)
            t += step

        # closed hold
        if self.stop_event.is_set():
            return
        self.set_blink(float(self.defaultintensity), float(self.defaultintensity))
        if self.send_callback:
            self.send_callback()

        # small micro-blinks (double blink) chance
        if random.random() < 0.12:
            # very short re-open then close again to simulate double blink
            self.set_blink(0.0, 0.0)
            if self.send_callback:
                self.send_callback()
            time.sleep(random.uniform(0.03, 0.08))
            # re-close quickly
            self.set_blink(float(self.defaultintensity), float(self.defaultintensity))
            if self.send_callback:
                self.send_callback()
            time.sleep(random.uniform(0.02, 0.06))

        if self.stop_event.is_set():
            return
        time.sleep(closed_hold)

        # opening (reverse of closing)
        t = 0.0
        while t < open_time:
            if self.stop_event.is_set():
                return
            p = self.ease_cos(min(1.0, 1.0 - (t / open_time)))
            self.set_blink( self.defaultintensity * p,
                            self.defaultintensity * max(0.0, min(1.0, p + asymmetry)))  
            if self.send_callback:
                self.send_callback()
            time.sleep(step)
            t += step

        # ensure fully open
        self.set_blink(0.0, 0.0)
        if self.send_callback:
            self.send_callback()


    def run(self):
        while not self.stop_event.is_set():
            # interval sampled from a skewed distribution (more realistic)
            interval = max(0.8, random.gauss(self.mean_interval, 1.2))
            # small chance of shorter interval (frequent blinks)
            if random.random() < 0.08:
                interval = random.uniform(1.0, 2.5)
            # wait but wake early if stop_event set
            if self.stop_event.wait(interval):
                break
            self.do_blink()

    def stop(self):
        self.stop_event.set()