package com.mangokeyboard

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
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

        buildSwitches()
        buildRows()
        tintWithAccent()

        buildExamples()
        watchField()
        loadModel()
    }

    override fun onResume() {
        super.onResume()
        readSwitches()
        updateValues()
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
        for (id in listOf(
            R.id.switch_auto_convert,
            R.id.switch_suggestions,
            R.id.switch_haptics,
            R.id.switch_completion,
            R.id.switch_word_list,
        )) {
            val switch = findViewById<Switch>(id)
            switch.thumbTintList = tint
            switch.trackTintList = tint
        }
    }

    // -------------------------------------------------------------- switches

    private fun buildSwitches() {
        findViewById<Switch>(R.id.switch_auto_convert).setOnCheckedChangeListener { _, value ->
            settings.autoConvert = value
        }
        findViewById<Switch>(R.id.switch_suggestions).setOnCheckedChangeListener { _, value ->
            settings.showSuggestions = value
        }
        findViewById<Switch>(R.id.switch_haptics).setOnCheckedChangeListener { _, value ->
            settings.haptics = value
        }
        findViewById<Switch>(R.id.switch_completion).setOnCheckedChangeListener { _, value ->
            settings.completion = value
        }
        findViewById<Switch>(R.id.switch_word_list).setOnCheckedChangeListener { _, value ->
            settings.wordList = value
        }
        readSwitches()
    }

    /** The state of the switches, from the settings. */
    private fun readSwitches() {
        findViewById<Switch>(R.id.switch_auto_convert).isChecked = settings.autoConvert
        findViewById<Switch>(R.id.switch_suggestions).isChecked = settings.showSuggestions
        findViewById<Switch>(R.id.switch_haptics).isChecked = settings.haptics
        findViewById<Switch>(R.id.switch_completion).isChecked = settings.completion
        findViewById<Switch>(R.id.switch_word_list).isChecked = settings.wordList
    }

    // ------------------------------------------------------------------ rows

    /** The rows that hold a value, and open a list when they are tapped. */
    private fun buildRows() {
        findViewById<View>(R.id.row_beam).setOnClickListener { showCandidates() }
        findViewById<View>(R.id.row_theme).setOnClickListener { showThemes() }
        findViewById<View>(R.id.row_accent).setOnClickListener { showColourPicker() }
        findViewById<View>(R.id.row_about).setOnClickListener {
            startActivity(Intent(this, AboutActivity::class.java))
        }
        updateValues()
    }

    /** The value on the right of every row that has one. */
    private fun updateValues() {
        findViewById<TextView>(R.id.value_beam).text = settings.beam.toString()
        findViewById<TextView>(R.id.value_theme).text =
            if (settings.darkMode) "Dark" else "Light"
        findViewById<TextView>(R.id.value_accent).text = accentName()
    }

    /** How many candidates the search keeps. */
    private fun showCandidates() {
        val options = arrayOf("1", "2", "4")
        showDialog(
            android.app.AlertDialog.Builder(this)
                .setTitle("Candidates")
                .setSingleChoiceItems(options, options.indexOf(settings.beam.toString())) { dialog, which ->
                    settings.beam = when (which) {
                        0 -> 1
                        1 -> 2
                        else -> 4
                    }
                    updateValues()
                    dialog.dismiss()
                }
                .setNegativeButton("Cancel", null),
        )
    }

    /** The theme of the keyboard and of the page. */
    private fun showThemes() {
        val options = arrayOf("Dark", "Light")
        showDialog(
            android.app.AlertDialog.Builder(this)
                .setTitle("Theme")
                .setSingleChoiceItems(options, if (settings.darkMode) 0 else 1) { dialog, which ->
                    settings.darkMode = which == 0
                    updateValues()
                    dialog.dismiss()
                    // The page is built with the colours of the theme, so it
                    // has to be built again when the theme changes.
                    recreate()
                }
                .setNegativeButton("Cancel", null),
        )
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
            label.text = "The Malayalam appears here"
            label.setTextColor(themeColour(R.attr.appFaint))
            return
        }
        label.text = word
        label.setTextColor(themeColour(R.attr.appText))
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

    /** The name of the accent colour, for the row that shows it. */
    private fun accentName(): String {
        for ((hex, name) in KeyboardSettings.ACCENTS) {
            if (Color.parseColor(hex) == settings.accent) return name
        }
        return "Custom"
    }

    private fun chip(): GradientDrawable {
        val shape = GradientDrawable()
        shape.setColor(themeColour(R.attr.appChip))
        shape.cornerRadius = 16f
        return shape
    }

    /**
     * The accent colour.
     *
     * The colours the app offers are at the top of the dialog, and a colour
     * that is not one of them can be made with the three sliders or by
     * writing its hex code.
     */
    private fun showColourPicker() {
        val panel = LinearLayout(this)
        panel.orientation = LinearLayout.VERTICAL
        panel.setPadding(dp(24), dp(20), dp(24), dp(8))

        // The colour as it is now.
        var current = settings.accent
        val preview = View(this)
        val previewParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, dp(56),
        )
        previewParams.topMargin = dp(16)
        previewParams.bottomMargin = dp(16)

        // The three channels and the hex code. They are filled in below, and
        // the listeners read them when they run.
        lateinit var red: SeekBar
        lateinit var green: SeekBar
        lateinit var blue: SeekBar
        lateinit var hex: EditText

        red = seekBar(Color.red(current)) {
            current = Color.rgb(it, Color.green(current), Color.blue(current))
            show(current, preview, red, green, blue, hex)
        }
        green = seekBar(Color.green(current)) {
            current = Color.rgb(Color.red(current), it, Color.blue(current))
            show(current, preview, red, green, blue, hex)
        }
        blue = seekBar(Color.blue(current)) {
            current = Color.rgb(Color.red(current), Color.green(current), it)
            show(current, preview, red, green, blue, hex)
        }

        // The hex code of the colour.
        hex = EditText(this)
        hex.setText(hexOf(current))
        hex.inputType = InputType.TYPE_CLASS_TEXT
        hex.imeOptions = EditorInfo.IME_ACTION_DONE
        hex.maxLines = 1
        hex.textSize = 16f
        hex.hint = "#2C5F8A"
        hex.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(text: Editable?) {
                val parsed = parseHex(text?.toString() ?: "") ?: return
                current = parsed
                show(current, preview, red, green, blue, hex)
            }

            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}

            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        })

        // The colours the app offers.
        val swatches = LinearLayout(this)
        swatches.orientation = LinearLayout.HORIZONTAL
        for ((code, name) in KeyboardSettings.ACCENTS) {
            val colour = Color.parseColor(code)
            swatches.addView(swatchButton(colour, name) {
                current = colour
                show(current, preview, red, green, blue, hex)
            })
        }

        panel.addView(swatches)
        panel.addView(preview, previewParams)
        preview.setBackgroundColor(current)
        panel.addView(red)
        panel.addView(green)
        panel.addView(blue)
        panel.addView(hex)

        showDialog(
            android.app.AlertDialog.Builder(this)
                .setTitle("Accent colour")
                .setView(panel)
                .setPositiveButton("Save") { _, _ ->
                    settings.accent = current
                    updateValues()
                    tintWithAccent()
                    Toast.makeText(this, "The keyboard will use it now", Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton("Cancel", null),
        )
    }

    /** Show a dialog whose buttons take the accent of the user. */
    private fun showDialog(builder: android.app.AlertDialog.Builder) {
        val dialog = builder.create()
        dialog.setOnShowListener {
            for (button in listOf(
                dialog.getButton(android.app.AlertDialog.BUTTON_POSITIVE),
                dialog.getButton(android.app.AlertDialog.BUTTON_NEGATIVE),
            )) {
                button?.setTextColor(settings.accent)
            }
        }
        dialog.show()
    }

    /** One of the colours the app offers, as a button. */
    private fun swatchButton(colour: Int, name: String, onClick: () -> Unit): View {
        val button = Button(this)
        button.contentDescription = name
        button.text = ""
        button.background = roundedSwatch(colour)
        button.setOnClickListener { onClick() }
        val params = LinearLayout.LayoutParams(0, dp(44), 1f)
        params.setMargins(0, 0, dp(6), 0)
        button.layoutParams = params
        return button
    }

    private fun roundedSwatch(colour: Int): GradientDrawable {
        val shape = GradientDrawable()
        shape.setColor(colour)
        shape.cornerRadius = 10f
        return shape
    }

    /** Put a colour into the preview, the sliders and the hex field. */
    private fun show(
        colour: Int,
        preview: View,
        red: SeekBar,
        green: SeekBar,
        blue: SeekBar,
        hex: EditText,
    ) {
        preview.setBackgroundColor(colour)
        if (red.progress != Color.red(colour)) red.progress = Color.red(colour)
        if (green.progress != Color.green(colour)) green.progress = Color.green(colour)
        if (blue.progress != Color.blue(colour)) blue.progress = Color.blue(colour)
        val text = hexOf(colour)
        if (hex.text.toString() != text) hex.setText(text)
    }

    /** The hex code of a colour, the way a colour picker writes it. */
    private fun hexOf(colour: Int): String = String.format("#%02X%02X%02X", Color.red(colour), Color.green(colour), Color.blue(colour))

    /** The colour of a hex code, or nothing when the code is not one. */
    private fun parseHex(text: String): Int? {
        val code = text.trim().removePrefix("#")
        if (code.length != 6) return null
        val value = code.toLongOrNull(16) ?: return null
        return Color.rgb(((value shr 16) and 0xFF).toInt(), ((value shr 8) and 0xFF).toInt(), (value and 0xFF).toInt())
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

    /** The version of the app, as it is in the package. */
    private fun versionName(): String {
        return try {
            val info = packageManager.getPackageInfo(packageName, 0)
            info.versionName ?: ""
        } catch (error: PackageManager.NameNotFoundException) {
            ""
        }
    }

    private fun dp(value: Int): Int = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP,
        value.toFloat(),
        resources.displayMetrics,
    ).toInt()
}
