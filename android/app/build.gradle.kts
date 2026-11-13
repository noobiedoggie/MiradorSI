plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.chaquo.python")
}

// ABI à compiler : `-Pabis=x86_64` pour l'émulateur seul (compilation native plus rapide).
val abis = (findProperty("abis") as String?)?.split(",") ?: listOf("arm64-v8a", "x86_64")

android {
    namespace = "com.miradorsi.app"
    compileSdk = 35
    // NDK déjà installé sur la machine (GitHub Actions en fournit un récent) ; sinon celui d'AGP
    System.getenv("ANDROID_NDK_LATEST_HOME")?.let { ndkVersion = it.trimEnd('/').substringAfterLast('/') }

    defaultConfig {
        applicationId = "com.miradorsi.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 4
        versionName = "0.4.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            // La quasi-totalité des téléphones récents ; x86_64 pour l'émulateur.
            abiFilters += abis
        }
        // Moteur natif des modèles ouverts (llama.cpp), toujours optimisé : en mode debug, il serait
        // des dizaines de fois plus lent. Bibliothèque C++ intégrée (c++_static) : Chaquopy livre
        // déjà un libc++_shared.so plus ancien, incompatible avec celui du NDK récent. Le moteur
        // n'échange que des types C entre ses bibliothèques (API C de llama.cpp et ggml).
        externalNativeBuild {
            cmake {
                arguments += listOf("-DCMAKE_BUILD_TYPE=Release", "-DANDROID_STL=c++_static")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        // Les variantes du moteur de calcul (selon le processeur) sont chargées depuis le dossier
        // des bibliothèques natives : elles doivent y être extraites à l'installation.
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

chaquopy {
    defaultConfig {
        version = "3.12"
        pip {
            install("numpy")
            install("certifi")
        }
        // mirador_app contient des fichiers de données (configuration, modèle, lexique) :
        // on l'extrait sur le stockage de l'application pour qu'ils soient lisibles normalement.
        extractPackages("mirador_app")
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation(platform("androidx.compose:compose-bom:2024.12.01"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
