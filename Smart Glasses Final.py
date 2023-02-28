from PIL import Image, ImageDraw, ImageFont
import speech_recognition as sr
import RPi.GPIO as GPIO
import adafruit_ssd1306
import numpy as np
import pytesseract
import digitalio
import pyttsx3
import board
import cv2
import re


def black_filter(frame):
    p = 0.3
    t1 = 135
    t2 = 105
    frame = frame.astype('float64')

    a1 = np.nan_to_num(frame[:, :, 0] / (frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2]), posinf=0, nan=0)
    a2 = frame[:, :, 1]
    a3 = frame[:, :, 2]

    a1 = a1 > p
    a2 = a2 < t1
    a3 = a3 < t2

    result = (a1.astype(np.uint8) * a2.astype(np.uint8) * a3.astype(np.uint8)) * 255

    return result


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.add_event_detect(17, GPIO.RISING, bouncetime = 250)
GPIO.add_event_detect(27, GPIO.RISING, bouncetime = 250)
GPIO.add_event_detect(22, GPIO.RISING, bouncetime = 250)

width = 128
height = 64

i2c = board.I2C()
oled = adafruit_ssd1306.SSD1306_I2C(width, height, i2c, addr=0x3c)
font = ImageFont.load_default()

image = Image.new("1", (oled.width, oled.height))
draw = ImageDraw.Draw(image)

for i in range(0, 3):
    oled.fill(1)
    oled.show()
    oled.fill(0)
    oled.show()

cap = cv2.VideoCapture(0)
engine = pyttsx3.init()
r = sr.Recognizer()

run_ocr = False
run_stt = False
debug_camera = False
f = None

while True:

    if GPIO.event_detected(17):
        print("Speech To Text Started!")
        run_stt = True

    elif GPIO.event_detected(27):
        print("OCR Started!")
        run_ocr = True

    elif GPIO.event_detected(22):
        print("Camera Debugging Started!")
        if debug_camera:
            cv2.destroyAllWindows()

        debug_camera = not debug_camera

    if debug_camera:
        t, f = cap.read()
        
        if f is None:
            continue
        
        filtered = black_filter(f)
        
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilation = cv2.dilate(filtered, rect_kernel, iterations=1)

        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True):
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Frame", f)
        cv2.imshow("Filter", filtered)
        cv2.waitKey(1)

        if f is None:
            continue

    if run_ocr:
        t, f = cap.read()
        
        if f is None:
            continue
        
        filtered = black_filter(f)
        
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilation = cv2.dilate(filtered, rect_kernel, iterations=1)

        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text = ""
        for cnt in sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0:2]:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = f[y:y + h, x:x + w]
        
            temp = pytesseract.image_to_string(cropped)
            text += " " + temp

        text = re.sub(r'[^\x00-\x7f]', r'', text).strip().replace("\n", " ")
        print(f"----------\n{text}\n----------")

        offset = 0
        s = ""
        for w in text.split(" "):
            font_width, font_height = font.getsize(s + " " + w)

            if font_width < 128:
                s += " " + w

            else:
                font_width, font_height = font.getsize(s)
                draw.text((width/2 - font_width/2, offset), s, fill="white", font=font)
                offset += 16
                s = w

        font_width, font_height = font.getsize(s)
        draw.text((width/2 - font_width/2, offset), s, fill="white", font=font)

        oled.image(image)
        oled.show()

        engine.say(text)
        engine.runAndWait()
        engine.stop()

        run_ocr = False

    if run_stt:

        with sr.Microphone() as s:
            r.adjust_for_ambient_noise(s, duration=4)

            v = r.listen(s)

            try:
                text = r.recognize_google(v)
                text = text.lower()

            except sr.RequestError or sr.UnknownValueError:
                text = r.recognize_sphinx(v)
                text = text.lower()

            print(f"----------\n{text}\n----------")

            offset = 0
            s = ""
            for w in text.split(" "):
                font_width, font_height = font.getsize(s + " " + w)

                if font_width < 128:
                    s += " " + w

                else:
                    font_width, font_height = font.getsize(s)
                    draw.text((width/2 - font_width/2, offset), s, fill="white", font=font)
                    offset += 20
                    s = w

            font_width, font_height = font.getsize(s)
            draw.text((width/2 - font_width/2, offset), s, fill="white", font=font)

            run_stt = False

cv2.destroyAllWindows()
GPIO.cleanup()