plugins {
    id("com.android.application") version "8.13.2"
    id("org.jetbrains.kotlin.android") version "2.2.10"
}

// The signing of the release. A key is read from the environment when one is
// there, and the debug key is used when there is none, so a prototype can
// always be published. The four variables are the four GitHub secrets in
// docs/RELEASING.md.
val keystorePath = System.getenv("MANGOKEYS_KEYSTORE")
val keystorePassword = System.getenv("MANGOKEYS_KEYSTORE_PASSWORD")
val keyAlias = System.getenv("MANGOKEYS_KEY_ALIAS")
val keyPassword = System.getenv("MANGOKEYS_KEY_PASSWORD")
val keystoreFile = keystorePath?.let { file(it) }
val signed = keystoreFile != null
    && keystoreFile.exists()
    && !keystorePassword.isNullOrEmpty()
    && !keyAlias.isNullOrEmpty()
    && !keyPassword.isNullOrEmpty()

android {
    namespace = "com.mangokeyboard"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.mangokeyboard"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        // The tag of the release sets this, so the page of the release and
        // the about section of the app always say the same thing.
        versionName = (project.findProperty("versionName") as String?) ?: "0.01"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    signingConfigs {
        if (signed) {
            create("release") {
                storeFile = keystoreFile
                storePassword = keystorePassword
                keyAlias = keyAlias
                keyPassword = keyPassword
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Without a key the release is signed with the debug key. It
            // installs on any phone, it just cannot be updated in place
            // later without the same key.
            signingConfig = if (signed) {
                signingConfigs.getByName("release")
            } else {
                signingConfigs.getByName("debug")
            }
        }
    }
}

// The model, the word list and the language model are copied into the
// package. The default is the committed bundle, so a build on a clean
// checkout works. The paths can be changed with -PmodelDir, -PlexiconDir
// and -PlexiconFile.
val modelDir: String = (project.findProperty("modelDir") as String?)
    ?: rootProject.file("../assets/release/model").path
val tokenizerFile: String = (project.findProperty("tokenizerFile") as String?)
    ?: rootProject.file("../assets/release/model/tokenizer.json").path
val lexiconDir: String = (project.findProperty("lexiconDir") as String?)
    ?: rootProject.file("../assets/release/lexicon").path
val ngramFile: String = (project.findProperty("ngramFile") as String?)
    ?: rootProject.file("../data/processed/ngram.txt").path
val prefixModelDir: String = (project.findProperty("prefixModelDir") as String?)
    ?: rootProject.file("../assets/release/prefix").path

val copyAssets by tasks.registering(Copy::class) {
    from(modelDir) {
        include("*.onnx")
    }
    from(tokenizerFile)
    from(lexiconDir) {
        into("lexicon")
    }
    if (file(ngramFile).exists()) {
        from(ngramFile)
    }
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
