package com.mangokeyboard

import android.app.Activity
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.mangokeyboard.transliterator.CharNgram
import com.mangokeyboard.transliterator.Keyboard
import com.mangokeyboard.transliterator.Lexicon
import java.io.File

/**
 * The application of MangoKeys.
 *
 * It has a test bench at the top, so that the keyboard can be tried without
 * switching the input method, and the settings of the keyboard under it.
 *
 * The page follows the theme the user chose, the same way the keyboard does.
 * The theme is set before the page is built, and the colours of the parts
 * that are made in code are read from it.
 */
class MainActivity : Activity() {

    private lateinit var settings: KeyboardSettings
    private val handler = Handler(Looper.getMainLooper())
    private var keyboard: Keyboard? = null
    private var worker: Thread? = null

    private val examples = listOf(
        "ente peru", "keralathil", "njangalkum", "sughamano", "aash", "bhashanam",
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = KeyboardSettings(this)
        setTheme(if (settings.darkMode) R.style.Theme_Manglish_Dark else R.style.Theme_Manglish)
        setContentView(R.layout.settings)

        findViewById<Switch>(R.id.switch_dark).apply {
            isChecked = settings.darkMode
            setOnCheckedChangeListener { _, value ->
                settings.darkMode = value
                // The page is built with the colours of the theme, so it has
                // to be built again when the theme changes.
                recreate()
            }
        }

        findViewById<Switch>(R.id.switch_completion).apply {
            isChecked = settings.completion
            setOnCheckedChangeListener { _, value -> settings.completion = value }
        }

        findViewById<Switch>(R.id.switch_word_list).apply {
            isChecked = settings.wordList
            setOnCheckedChangeListener { _, value -> settings.wordList = value }
        }

        val group = findViewById<android.widget.RadioGroup>(R.id.group_beam)
        group.check(
            when (settings.beam) {
                1 -> R.id.beam_1
                2 -> R.id.beam_2
                else -> R.id.beam_4
            },
        )
        group.setOnCheckedChangeListener { _, id ->
            settings.beam = when (id) {
                R.id.beam_1 -> 1
                R.id.beam_2 -> 2
                else -> 4
            }
        }

        tintWithAccent()

        buildAccentRow()
        findViewById<Button>(R.id.button_custom_accent).setOnClickListener { showColourPicker() }
        buildExamples()
        watchField()
        loadModel()

        findViewById<TextView>(R.id.about).text = """
            The transliteration is done by two character level models of 5.4 million
            parameters each, and they run on the phone. The first one answers whole
            words. The second one was trained on the same words plus the pairs where
            the roman side is cut short, so it knows where the word is going, and the
            word list completes it.

            The models were trained on the Aksharantar corpus of AI4Bharat and the
            word list comes from the frequency list of the OpenSubtitles corpus and
            from the Malayalam lexicon of the Dakshina dataset of Google.
        """.trimIndent()
    }

    override fun onResume() {
        super.onResume()
        findViewById<Switch>(R.id.switch_dark).isChecked = settings.darkMode
        findViewById<Switch>(R.id.switch_completion).isChecked = settings.completion
        findViewById<Switch>(R.id.switch_word_list).isChecked = settings.wordList
        buildAccentRow()
    }

    // ----------------------------------------------------------------- theme

    /** One of the colours of the current theme. */
    private fun themeColour(attribute: Int): Int {
        val typed = TypedValue()
        theme.resolveAttribute(attribute, typed, true)
        return if (typed.resourceId != 0) {
            ContextCompat.getColor(this, typed.resourceId)
        } else {
            typed.data
        }
    }

    /** The switches and the radio buttons take the accent of the user. */
    private fun tintWithAccent() {
        val states = arrayOf(
            intArrayOf(android.R.attr.state_checked),
            intArrayOf(-android.R.attr.state_checked),
        )
        val colors = intArrayOf(settings.accent, themeColour(R.attr.appFaint))
        val tint = ColorStateList(states, colors)
        for (id in listOf(R.id.switch_dark, R.id.switch_completion, R.id.switch_word_list)) {
            val switch = findViewById<Switch>(id)
            switch.thumbTintList = tint
            switch.trackTintList = tint
        }
        for (id in listOf(R.id.beam_1, R.id.beam_2, R.id.beam_4)) {
            findViewById<RadioButton>(id).buttonTintList = tint
        }
    }

    // ------------------------------------------------------------- test bench

    /** The model, loaded in the background so the page comes up straight away. */
    private fun loadModel() {
        if (keyboard != null) return
        Thread {
            val models = ModelAssets(this).directory
            val lexicon = if (settings.wordList) {
                Lexicon().load(File(models, "lexicon"))
            } else {
                Lexicon(emptyArray(), IntArray(0))
            }
            val ngramFile = File(models, "ngram.txt")
            val ngram = if (ngramFile.exists()) CharNgram.load(ngramFile) else null
            val prefix = File(models, "prefix")
            val engine = Keyboard(
                models,
                File(models, "tokenizer.json").readText(),
                lexicon,
                ngram,
                prefixModel = if (prefix.exists() && settings.completion) prefix else null,
                beam = settings.beam,
            )
            handler.post { keyboard = engine; translate() }
        }.also { it.start() }
    }

    private fun watchField() {
        val field = findViewById<EditText>(R.id.field)
        field.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(text: Editable?) = translate()
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        })
    }

    /** Show what the last word of the field would become. */
    private fun translate() {
        val field = findViewById<EditText>(R.id.field)
        val text = field.text.toString()
        val word = text.split(Regex("[\\s,.]")).lastOrNull { it.isNotEmpty() } ?: ""
        val label = findViewById<TextView>(R.id.translation)
        if (word.isEmpty()) {
            label.text = "…"
            return
        }
        label.text = word
        val engine = keyboard ?: return
        worker?.interrupt()
        worker = Thread {
            val ranked = engine.suggest(word)
            val text = if (ranked.isEmpty()) word else ranked.joinToString("   ")
            handler.post {
                if (findViewById<EditText>(R.id.field).text.toString().contains(word)) {
                    label.text = text
                }
            }
        }.also { it.start() }
    }

    private fun buildExamples() {
        val row = findViewById<LinearLayout>(R.id.examples)
        row.removeAllViews()
        for (example in examples) {
            val button = Button(this)
            button.text = example
            button.textSize = 13f
            button.isAllCaps = false
            button.setTextColor(themeColour(R.attr.appMuted))
            button.background = chip()
            button.setOnClickListener {
                val field = findViewById<EditText>(R.id.field)
                field.setText(example)
                field.setSelection(example.length)
            }
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            )
            params.setMargins(0, 0, dp(8), 0)
            button.layoutParams = params
            row.addView(button)
        }
    }

    // ------------------------------------------------------------- appearance

    /** One round button for every colour the settings page offers. */
    private fun buildAccentRow() {
        val row = findViewById<LinearLayout>(R.id.accent_row)
        row.removeAllViews()
        for ((hex, name) in KeyboardSettings.ACCENTS) {
            val colour = Color.parseColor(hex)
            val button = Button(this)
            button.text = name
            button.textSize = 12f
            button.isAllCaps = false
            button.setTextColor(KeyboardSettings.readableOn(colour))
            button.background = swatch(colour, settings.accent == colour)
            button.setOnClickListener {
                settings.accent = colour
                buildAccentRow()
                tintWithAccent()
            }
            val params = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            params.setMargins(0, 0, dp(6), 0)
            button.layoutParams = params
            row.addView(button)
        }
    }

    private fun swatch(colour: Int, selected: Boolean): GradientDrawable {
        val shape = GradientDrawable()
        shape.setColor(colour)
        shape.cornerRadius = 12f
        val edge = if (selected) themeColour(R.attr.appText) else shade(colour)
        shape.setStroke(if (selected) 3 else 1, edge)
        return shape
    }

    private fun chip(): GradientDrawable {
        val shape = GradientDrawable()
        shape.setColor(themeColour(R.attr.appChip))
        shape.cornerRadius = 16f
        return shape
    }

    private fun shade(colour: Int): Int = Color.rgb(
        (Color.red(colour) * 0.8).toInt(),
        (Color.green(colour) * 0.8).toInt(),
        (Color.blue(colour) * 0.8).toInt(),
    )

    /** A small dialog with three sliders, one for each channel. */
    private fun showColourPicker() {
        val colour = settings.accent
        val panel = LinearLayout(this)
        panel.orientation = LinearLayout.VERTICAL
        panel.setPadding(dp(28), dp(20), dp(28), dp(8))

        val preview = View(this)
        val previewParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, dp(56),
        )
        previewParams.bottomMargin = dp(12)
        panel.addView(preview, previewParams)

        val red = seekBar(Color.red(colour)) { preview.setBackgroundColor(colour(it, 0, 0)) }
        val green = seekBar(Color.green(colour)) { preview.setBackgroundColor(colour(0, it, 0)) }
        val blue = seekBar(Color.blue(colour)) { preview.setBackgroundColor(colour(0, 0, it)) }
        panel.addView(red)
        panel.addView(green)
        panel.addView(blue)
        preview.setBackgroundColor(colour)

        android.app.AlertDialog.Builder(this)
            .setTitle("Accent colour")
            .setView(panel)
            .setPositiveButton("Save") { _, _ ->
                settings.accent = Color.rgb(red.progress, green.progress, blue.progress)
                buildAccentRow()
                tintWithAccent()
                Toast.makeText(this, "The keyboard will use it now", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun seekBar(value: Int, onChange: (Int) -> Unit): SeekBar {
        val bar = SeekBar(this)
        bar.max = 255
        bar.progress = value
        bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seek: SeekBar, progress: Int, fromUser: Boolean) {
                onChange(progress)
            }

            override fun onStartTrackingTouch(seek: SeekBar) {}

            override fun onStopTrackingTouch(seek: SeekBar) {}
        })
        return bar
    }

    private fun colour(red: Int, green: Int, blue: Int): Int = Color.rgb(red, green, blue)

    private fun dp(value: Int): Int = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP,
        value.toFloat(),
        resources.displayMetrics,
    ).toInt()
}
