package com.mangokeyboard

import android.app.Activity
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast

/**
 * The settings of the keyboard.
 *
 * The appearance is the theme and the accent colour. The behaviour is what
 * the keyboard does with the word list and with the second model.
 */
class SettingsActivity : Activity() {

    private lateinit var settings: KeyboardSettings

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = KeyboardSettings(this)
        setContentView(R.layout.settings)

        findViewById<Switch>(R.id.switch_dark).apply {
            isChecked = settings.darkMode
            setOnCheckedChangeListener { _, value -> settings.darkMode = value }
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
        val chosen = when (settings.beam) {
            1 -> R.id.beam_1
            2 -> R.id.beam_2
            else -> R.id.beam_4
        }
        group.check(chosen)
        group.setOnCheckedChangeListener { _, id ->
            settings.beam = when (id) {
                R.id.beam_1 -> 1
                R.id.beam_2 -> 2
                else -> 4
            }
        }

        buildAccentRow()
        findViewById<Button>(R.id.button_custom_accent).setOnClickListener {
            showColourPicker()
        }

        findViewById<TextView>(R.id.about).text = """
            The transliteration is done by two character level models of 5.4 million
            parameters each, and they run on the phone. The first one answers whole
            words. The second one was trained on the same words plus the pairs where
            the roman side is cut short, so it knows where the word is going, and the
            word list completes it.

            Nothing you type leaves the phone. The keyboard asks for no permissions
            at all.
        """.trimIndent()
    }

    override fun onResume() {
        super.onResume()
        findViewById<Switch>(R.id.switch_dark).isChecked = settings.darkMode
        findViewById<Switch>(R.id.switch_completion).isChecked = settings.completion
        findViewById<Switch>(R.id.switch_word_list).isChecked = settings.wordList
        buildAccentRow()
    }

    /** One round button for every colour the settings page offers. */
    private fun buildAccentRow() {
        val row = findViewById<LinearLayout>(R.id.accent_row)
        row.removeAllViews()
        for ((hex, name) in KeyboardSettings.ACCENTS) {
            val colour = Color.parseColor(hex)
            val button = Button(this)
            button.text = name
            button.textSize = 12f
            button.setTextColor(KeyboardSettings.readableOn(colour))
            button.isAllCaps = false
            button.background = swatch(colour, settings.accent == colour)
            button.setOnClickListener {
                settings.accent = colour
                buildAccentRow()
            }
            val params = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            params.setMargins(0, 0, 8, 0)
            button.layoutParams = params
            row.addView(button)
        }
    }

    private fun swatch(colour: Int, selected: Boolean): GradientDrawable {
        val shape = GradientDrawable()
        shape.setColor(colour)
        shape.cornerRadius = 14f
        val border = if (selected) 3 else 1
        shape.setStroke(border, if (selected) Color.parseColor("#111827") else shade(colour))
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
        panel.setPadding(32, 24, 32, 8)

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

        val dialog = android.app.AlertDialog.Builder(this)
            .setTitle("Accent colour")
            .setView(panel)
            .setPositiveButton("Save") { _, _ ->
                settings.accent = Color.rgb(red.progress, green.progress, blue.progress)
                buildAccentRow()
                Toast.makeText(this, "The keyboard will use it now", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .create()
        dialog.show()
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

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
}
