plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "ups.com.emotionrecognitionapp"
    compileSdk = 34

    defaultConfig {
        applicationId = "ups.com.emotionrecognitionapp"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    // Importa la BoM de Firebase
    implementation(platform("com.google.firebase:firebase-bom:33.0.0"))

    // Agrega las dependencias de Firebase que necesites
    // Por ejemplo:
    // implementation("com.google.firebase:firebase-analytics")

    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}