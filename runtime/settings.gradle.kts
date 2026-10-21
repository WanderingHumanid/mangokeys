pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "mangokeyboard"

include(":android")
project(":android").projectDir = file("android")

include(":app")
project(":app").projectDir = file("../apps/android")
