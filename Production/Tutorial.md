# Introduction
Today, there is rising use and demand of mobile devices among users due to their potability and ease of use and there is need to utilize this devices to build Artificial intelligence apps. In this tutorial we are going to learn how to deploy a speaker recognition model on android device which is a domain of speech recognition. When you check your phone now there are voice assistant apps like Google assistant and Siri which are powered by AI.

Am trusting you have gone through the Python tutorial on how to train a speaker recognition model of five speakers if not please do consider visiting it [here]().

# Tutorial Overview
This tutorial is divided into 5 sections namely:
1. Setting up dependencies and assets, 
2. Building recognition view,
3. Handling user permissions,
4. Building speaker activity,
5. Building recognition activity, and
6. Running the app

# 1. Setting up dependencies and assets
Most of the times we will be relying on dependencies to accomplish several tasks with the crucial dependencies being TensorFlow Lite and TensorFlow Support which will help us read models and perform tensor operations.

From the development post we saved two important asset files (i.e. tflite_model.tflite and labels.txt) former being our model to perform predictions and latter ordered names of the speakers. Create assets folder on android main folder and place this two asset files.

Open build.gradle(Module:app) and add the following lines. 
```
// set no compress for tflite

compileOptions {
    sourceCompatibility JavaVersion.VERSION_1_8
    targetCompatibility JavaVersion.VERSION_1_8
}

android {
    aaptOptions {
        noCompress "tflite"
    }
}

dependencies {
    // TensorFlow Lite dependency
    implementation "org.tensorflow:tensorflow-lite:2.1.0"
    // Support library
    implementation('org.tensorflow:tensorflow-lite-support:0.0.0-nightly') { changing = true }

}

```

# 2. Building recognition view
To see which speaker has been recognized, we will build a view with their names on it where will change the background of recognized speaker for visual representation of the result. Will make use of constrained layout and text views and will look like this;

# 3. Handling user permissions
For security purposes, it's a good practice to have the user make a choice on whether to allow or deny app permissions to use certain features and for our case recording audio.

# 4. Building speaker activity
We need to record speaker voice from the mic and here we will handle events to record audio from the mic. The audio duration to be recorded if of length 1000 milliseconds and with a the following properties; sample rate of 16000, mono channel and 16bit PCM encoded.


