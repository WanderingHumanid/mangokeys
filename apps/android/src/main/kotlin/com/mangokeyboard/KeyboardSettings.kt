package com.mangokeyboard

import android.content.Context
import android.graphics.Color

/** The settings of the keyboard, shared between the settings page and the keyboard. */
class KeyboardSettings(context: Context) {

    private val prefs = context.getSharedPreferences(NAME, Context.MODE_PRIVATE)

    /** The dark theme. */
    var darkMode: Boolean
        get() = prefs.getBoolean(KEY_DARK, false)
        set(value) = prefs.edit().putBoolean(KEY_DARK, value).apply()

    /** The accent colour of the keyboard. */
    var accent: Int
        get() = prefs.getInt(KEY_ACCENT, DEFAULT_ACCENT)
        set(value) = prefs.edit().putInt(KEY_ACCENT, value).apply()

    /** Whether the space bar converts the word that is being typed. */
    var autoConvert: Boolean
        get() = prefs.getBoolean(KEY_AUTO_CONVERT, true)
        set(value) = prefs.edit().putBoolean(KEY_AUTO_CONVERT, value).apply()

    /** Whether the keys vibrate when they are pressed. */
    var haptics: Boolean
        get() = prefs.getBoolean(KEY_HAPTICS, true)
        set(value) = prefs.edit().putBoolean(KEY_HAPTICS, value).apply()

    /** Whether the candidates are shown above the keys. */
    var showSuggestions: Boolean
        get() = prefs.getBoolean(KEY_SUGGESTIONS, true)
        set(value) = prefs.edit().putBoolean(KEY_SUGGESTIONS, value).apply()

    /** Whether the prefix model may complete the word. */
    var completion: Boolean
        get() = prefs.getBoolean(KEY_COMPLETION, true)
        set(value) = prefs.edit().putBoolean(KEY_COMPLETION, value).apply()

    /** Whether the word list may be used. */
    var wordList: Boolean
        get() = prefs.getBoolean(KEY_WORD_LIST, true)
        set(value) = prefs.edit().putBoolean(KEY_WORD_LIST, value).apply()

    /** How many candidates the search keeps. */
    var beam: Int
        get() = prefs.getInt(KEY_BEAM, 4)
        set(value) = prefs.edit().putInt(KEY_BEAM, value).apply()

    /** The colours of the keyboard for the current theme. */
    fun palette(): Palette = if (darkMode) {
        Palette(
            background = Color.parseColor("#1B1D21"),
            panel = Color.parseColor("#23262B"),
            key = Color.parseColor("#31353B"),
            function = Color.parseColor("#3A3F46"),
            keyText = Color.parseColor("#E8EAED"),
            functionText = Color.parseColor("#C4C8CE"),
            panelText = Color.parseColor("#E8EAED"),
            accent = accent,
            accentText = readableOn(accent),
        )
    } else {
        Palette(
            background = Color.parseColor("#EDEFF2"),
            panel = Color.parseColor("#F7F8FA"),
            key = Color.parseColor("#FFFFFF"),
            function = Color.parseColor("#DFE2E7"),
            keyText = Color.parseColor("#1A1C1E"),
            functionText = Color.parseColor("#3C4043"),
            panelText = Color.parseColor("#1A1C1E"),
            accent = accent,
            accentText = readableOn(accent),
        )
    }

    companion object {
        const val NAME = "mangokeys"

        const val KEY_DARK = "dark_mode"
        const val KEY_ACCENT = "accent"
        const val KEY_AUTO_CONVERT = "auto_convert"
        const val KEY_SUGGESTIONS = "suggestions"
        const val KEY_HAPTICS = "haptics"
        const val KEY_COMPLETION = "completion"
        const val KEY_WORD_LIST = "word_list"
        const val KEY_BEAM = "beam"

        val DEFAULT_ACCENT = Color.parseColor("#2C5F8A")

        /** The colours that the settings page offers. */
        val ACCENTS = listOf(
            "#2C5F8A" to "Ocean",
            "#7A2E8F" to "Grape",
            "#1D6B4F" to "Forest",
            "#B3541E" to "Ember",
            "#A61E4D" to "Berry",
            "#4A5568" to "Slate",
        )

        /** A colour that can be read on the accent colour. */
        fun readableOn(color: Int): Int {
            val red = Color.red(color) * 0.299
            val green = Color.green(color) * 0.587
            val blue = Color.blue(color) * 0.114
            return if (red + green + blue > 165) Color.parseColor("#1A1C1E") else Color.WHITE
        }
    }
}

/** The colours of one theme. */
data class Palette(
    val background: Int,
    val panel: Int,
    val key: Int,
    val function: Int,
    val keyText: Int,
    val functionText: Int,
    val panelText: Int,
    val accent: Int,
    val accentText: Int,
)
