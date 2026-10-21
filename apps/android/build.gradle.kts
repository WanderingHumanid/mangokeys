plugins {
    id("com.android.application") version "8.13.2"
    id("org.jetbrains.kotlin.android") version "2.2.10"
}

android {
    namespace = "com.mangokeyboard"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.mangokeyboard"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
}

// The model, the word list and the language model are copied into the
// package. The paths can be changed with -PmodelDir, -PlexiconDir and
// -PlexiconFile.
val modelDir: String = (project.findProperty("modelDir") as String?)
    ?: rootProject.file("../runs/demo/onnx-int8").path
val tokenizerFile: String = (project.findProperty("tokenizerFile") as String?)
    ?: rootProject.file("../runs/ml_mini/tokenizer.json").path
val lexiconDir: String = (project.findProperty("lexiconDir") as String?)
    ?: rootProject.file("../data/processed/lexicon").path
val ngramFile: String = (project.findProperty("ngramFile") as String?)
    ?: rootProject.file("../data/processed/ngram.txt").path
val prefixModelDir: String = (project.findProperty("prefixModelDir") as String?)
    ?: rootProject.file("../runs/prefix/onnx-int8").path

val copyAssets by tasks.registering(Copy::class) {
    from(modelDir) {
        include("*.onnx")
    }
    from(tokenizerFile)
    from(lexiconDir) {
        into("lexicon")
    }
    from(ngramFile)
    from(prefixModelDir) {
        include("*.onnx")
        rename { "prefix-${it}" }
    }
    into(layout.buildDirectory.dir("generated/assets"))
}

android.sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/assets"))

tasks.named("preBuild") {
    dependsOn(copyAssets)
}

dependencies {
    implementation(project(":android"))
    implementation("androidx.core:core-ktx:1.13.1")
}
