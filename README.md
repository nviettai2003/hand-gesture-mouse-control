## 🧱 Yocto Environment

### Yocto Release
- **Kirkstone**

### Layers Used
- `poky`
- `meta-openembedded`
- `meta-raspberrypi`
- `meta-tensorflow-lite`

### Image & Configuration
- **Target Image**: `core-image-sato`
- **Build Configuration** (`conf/local.conf`):
    - `IMAGE_INSTALL:append = " opencv libcamera libcamera-dev libtensorflow-lite"`
    - `EXTRA_OEMESON:append:pn-libcamera = " -Dpipelines=raspberrypi -Dipas=raspberrypi"`

## 🛠 Technical Stack
* **Hardware**: **Raspberry Pi 4B**  **Camera Module v2 (Sony IMX219)**
* **Core Language**: **C++** 
* **Camera Framework**: **libcamera** 
* **Computer Vision**: **OpenCV** 
* **AI Engine**: **TensorFlow Lite**

## 🎥 Demo
https://github.com/user-attachments/assets/1f460b73-7061-4154-8dcb-68830c886ae9

