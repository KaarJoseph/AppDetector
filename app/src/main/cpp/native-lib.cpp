#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <android/bitmap.h>
#include <android/log.h>
#include <sstream>
#include <vector>

#define TAG "native-lib"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using namespace cv;

void bitmapToMat(JNIEnv *env, jobject bitmap, Mat &dst) {
    AndroidBitmapInfo info;
    void* pixels = 0;

    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 || info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);

    dst.create(info.height, info.width, CV_8UC4);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        Mat tmp(info.height, info.width, CV_8UC4, pixels);
        tmp.copyTo(dst);
    } else {
        Mat tmp(info.height, info.width, CV_8UC2, pixels);
        cvtColor(tmp, dst, COLOR_BGR5652RGBA);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}

void matToBitmap(JNIEnv *env, Mat &src, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels = 0;

    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 || info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        Mat tmp(info.height, info.width, CV_8UC4, pixels);
        if (src.type() == CV_8UC1) {
            cvtColor(src, tmp, COLOR_GRAY2BGRA);
        } else if (src.type() == CV_8UC3) {
            cvtColor(src, tmp, COLOR_RGB2BGRA);
        } else if (src.type() == CV_8UC4) {
            src.copyTo(tmp);
        }
    } else {
        Mat tmp(info.height, info.width, CV_8UC2, pixels);
        if (src.type() == CV_8UC1) {
            cvtColor(src, tmp, COLOR_GRAY2BGR565);
        } else if (src.type() == CV_8UC3) {
            cvtColor(src, tmp, COLOR_RGB2BGR565);
        } else if (src.type() == CV_8UC4) {
            cvtColor(src, tmp, COLOR_RGBA2BGR565);
        }
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_ups_com_emotionrecognitionapp_MainActivity_detectFeatures(JNIEnv *env, jobject instance, jobject bitmap, jobject processedBitmap) {
    Mat img;
    bitmapToMat(env, bitmap, img);

    if (img.empty()) {
        LOGE("Image is empty");
        return env->NewStringUTF("");
    }

    Mat imgProcessed = img.clone(); // Crear una copia para el procesamiento

    std::string faceCascadePath = "/data/user/0/ups.com.emotionrecognitionapp/files/haarcascade_frontalface_alt.xml";
    std::string eyeCascadePath = "/data/user/0/ups.com.emotionrecognitionapp/files/haarcascade_eye.xml";
    std::string noseCascadePath = "/data/user/0/ups.com.emotionrecognitionapp/files/haarcascade_mcs_nose.xml";
    std::string mouthCascadePath = "/data/user/0/ups.com.emotionrecognitionapp/files/haarcascade_mcs_mouth.xml";

    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade;
    CascadeClassifier noseCascade;
    CascadeClassifier mouthCascade;

    if (!faceCascade.load(faceCascadePath) ||
        !eyeCascade.load(eyeCascadePath) ||
        !noseCascade.load(noseCascadePath) ||
        !mouthCascade.load(mouthCascadePath)) {
        LOGE("Error loading cascade files");
        return env->NewStringUTF("");
    }

    std::vector<Rect> faces;
    faceCascade.detectMultiScale(img, faces);

    std::vector<Rect> eyes, noses, mouths;
    std::stringstream ss;
    ss << "{";

    for (size_t i = 0; i < faces.size(); i++) {
        rectangle(imgProcessed, faces[i], Scalar(255, 0, 0), 2);
        Mat faceROI = img(faces[i]);
        eyeCascade.detectMultiScale(faceROI, eyes);
        noseCascade.detectMultiScale(faceROI, noses);
        mouthCascade.detectMultiScale(faceROI, mouths);

        ss << "\"face\": {\"x\": " << faces[i].x << ", \"y\": " << faces[i].y
           << ", \"width\": " << faces[i].width << ", \"height\": " << faces[i].height << "}, ";

        ss << "\"eyes\": [";
        for (size_t j = 0; j < eyes.size(); j++) {
            Rect eye = eyes[j];
            eye.x += faces[i].x;
            eye.y += faces[i].y;
            rectangle(imgProcessed, eye, Scalar(0, 255, 0), 2);
            if (j > 0) ss << ", ";
            ss << "{\"x\": " << eye.x << ", \"y\": " << eye.y
               << ", \"width\": " << eye.width << ", \"height\": " << eye.height << "}";
        }
        ss << "], \"noses\": [";
        for (size_t j = 0; j < noses.size(); j++) {
            Rect noseRect = noses[j];
            noseRect.x += faces[i].x;
            noseRect.y += faces[i].y;
            rectangle(imgProcessed, noseRect, Scalar(0, 255, 255), 2);
            if (j > 0) ss << ", ";
            ss << "{\"x\": " << noseRect.x << ", \"y\": " << noseRect.y
               << ", \"width\": " << noseRect.width << ", \"height\": " << noseRect.height << "}";
        }
        ss << "], \"mouths\": [";
        for (size_t j = 0; j < mouths.size(); j++) {
            Rect mouthRect = mouths[j];
            mouthRect.x += faces[i].x;
            mouthRect.y += faces[i].y;
            rectangle(imgProcessed, mouthRect, Scalar(255, 0, 255), 2);
            if (j > 0) ss << ", ";
            ss << "{\"x\": " << mouthRect.x << ", \"y\": " << mouthRect.y
               << ", \"width\": " << mouthRect.width << ", \"height\": " << mouthRect.height << "}";
        }
        ss << "]";
    }
    ss << "}";

    matToBitmap(env, imgProcessed, processedBitmap);

    std::string resultJson = ss.str();

    LOGE("Result JSON: %s", resultJson.c_str());

    return env->NewStringUTF(resultJson.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_ups_com_emotionrecognitionapp_MainActivity_calculateHOG(JNIEnv *env, jobject instance, jobject bitmap) {
    Mat img;
    bitmapToMat(env, bitmap, img);

    if (img.channels() == 4) {
        cvtColor(img, img, COLOR_RGBA2GRAY);
    } else if (img.channels() == 3) {
        cvtColor(img, img, COLOR_RGB2GRAY);
    }

    HOGDescriptor hog;
    std::vector<float> descriptors;
    hog.compute(img, descriptors);

    std::ostringstream hogStr;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        hogStr << descriptors[i] << " ";
    }

    LOGE("HOG Descriptors: %s", hogStr.str().c_str());

    return env->NewStringUTF(hogStr.str().c_str());
}
