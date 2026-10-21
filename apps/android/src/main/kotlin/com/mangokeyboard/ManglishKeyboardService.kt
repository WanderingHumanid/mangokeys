package com.mangokeyboard

import android.inputmethodservice.InputMethodService
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.LinearLayout
import android.widget.TextView
import com.mangokeyboard.transliterator.CharNgram
import com.mangokeyboard.transliterator.Keyboard
import com.mangokeyboard.transliterator.Lexicon
import java.io.File

/**
 * The MangoKeys keyboard.
 *
 * It behaves like an ordinary keyboard. Every key goes straight into the text
 * field that has the focus, so the backspace, the cursor and the selection
 * are the ones of the application, not of the keyboard. What the keyboard
 * adds is the suggestion strip: the word the user is typing is translated and
 * the results sit above the keys, and tapping one replaces the typed word in
 * the field with the Malayalam.
 *
 * The model is loaded once, when the service is created. The onnx runtime
 * needs file paths, so the assets are copied into the private directory of
 * the application first.
 */
class ManglishKeyboardService : InputMethodService() {

    private lateinit var keyboard: Keyboard
    private lateinit var view: View
    private lateinit var strip: LinearLayout
    private lateinit var keys: LinearLayout
    private lateinit var settings: KeyboardSettings
    private val handler = Handler(Looper.getMainLooper())
    private var worker: Thread? = null
    private var beam = 4
    private var completion = true
    private var wordList = true
    private var shift = KeyboardLayout.Shift.OFF
    private var numbers = false
    private var palette = Palette(
        background = android.graphics.Color.parseColor("#EDEFF2"),
        panel = android.graphics.Color.parseColor("#F7F8FA"),
        key = android.graphics.Color.WHITE,
        function = android.graphics.Color.parseColor("#DFE2E7"),
        keyText = android.graphics.Color.parseColor("#1A1C1E"),
        functionText = android.graphics.Color.parseColor("#3C4043"),
        panelText = android.graphics.Color.parseColor("#1A1C1E"),
        accent = KeyboardSettings.DEFAULT_ACCENT,
        accentText = android.graphics.Color.WHITE,
    )

    override fun onCreate() {
        super.onCreate()
        settings = KeyboardSettings(this)
        palette = settings.palette()
        buildKeyboard()
        Log.i(TAG, "the model is ready")
    }

    /** Build the model. The settings can change it, so it is built again. */
    private fun buildKeyboard() {
        val models = ModelAssets(this).directory
        val lexicon = if (settings.wordList) {
            Lexicon().load(File(models, "lexicon"))
        } else {
            Lexicon(emptyArray(), IntArray(0))
        }
        val ngramFile = File(models, "ngram.txt")
        val ngram = if (ngramFile.exists()) CharNgram.load(ngramFile) else null
        val prefix = File(models, "prefix")
        keyboard = Keyboard(
            models,
            File(models, "tokenizer.json").readText(),
            lexicon,
            ngram,
            prefixModel = if (prefix.exists() && settings.completion) prefix else null,
            beam = settings.beam,
        )
    }

    override fun onCreateInputView(): View {
        view = layoutInflater.inflate(R.layout.keyboard, null)
        strip = view.findViewById(R.id.suggestion_strip)
        keys = view.findViewById(R.id.keys)
        buildKeys()
        applyTheme()
        return view
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        // The settings may have changed since the last time.
        if (settings.beam != beam
            || settings.completion != completion
            || settings.wordList != wordList
        ) {
            beam = settings.beam
            completion = settings.completion
            wordList = settings.wordList
            buildKeyboard()
        }
        palette = settings.palette()
        shift = KeyboardLayout.Shift.OFF
        numbers = false
        buildKeys()
        applyTheme()
        refresh()
    }

    override fun onFinishInput() {
        super.onFinishInput()
        show(emptyList())
    }

    // ------------------------------------------------------------------ keys

    private fun buildKeys() {
        KeyboardLayout.build(
            this,
            keys,
            palette,
            shift,
            numbers,
            onKey = { type(it) },
            onShift = { cycleShift() },
            onNumbers = { numbers = true; buildKeys() },
            onLetters = { numbers = false; buildKeys() },
            onBackspace = { backspace() },
            onEnter = { enter() },
            onGlobe = { switchKeyboard() },
        )
    }

    private fun cycleShift() {
        shift = when (shift) {
            KeyboardLayout.Shift.OFF -> KeyboardLayout.Shift.ON
            KeyboardLayout.Shift.ON -> KeyboardLayout.Shift.LOCKED
            KeyboardLayout.Shift.LOCKED -> KeyboardLayout.Shift.OFF
        }
        buildKeys()
    }

    // ------------------------------------------------------------------ input

    /**
     * The word the cursor is in, in the text field that has the focus.
     *
     * Only the Roman letters count. The Malayalam that the keyboard wrote
     * earlier is not a word being typed, and the suggestions must not be
     * about it.
     */
    private fun currentWord(): String {
        val before = currentInputConnection?.getTextBeforeCursor(64, 0)?.toString() ?: return ""
        var start = before.length
        while (start > 0 && isWordChar(before[start - 1])) start--
        return before.substring(start)
    }

    private fun isWordChar(char: Char): Boolean =
        char in 'a'..'z' || char in 'A'..'Z' || char == '\'' || char == '-'

    private fun type(letter: String) {
        val text = when (shift) {
            KeyboardLayout.Shift.LOCKED -> letter
            KeyboardLayout.Shift.ON -> letter.uppercase()
            KeyboardLayout.Shift.OFF -> letter
        }
        if (shift == KeyboardLayout.Shift.ON) {
            shift = KeyboardLayout.Shift.OFF
            buildKeys()
        }
        currentInputConnection?.commitText(text, 1)
        refresh()
    }

    private fun backspace() {
        val connection = currentInputConnection ?: return
        connection.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_DEL))
        connection.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_DEL))
        refresh()
    }

    private fun enter() {
        val connection = currentInputConnection ?: return
        connection.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER))
        connection.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER))
        refresh()
    }

    /** The space bar commits the best suggestion and then a space. */
    private fun space() {
        val connection = currentInputConnection ?: return
        val typed = currentWord()
        val best = firstSuggestion()
        if (!typed.isEmpty() && best != null) {
            // The space bar leaves the space itself, so the word does not
            // need one.
            replaceWord(best, trailingSpace = false)
        }
        connection.commitText(" ", 1)
        refresh()
    }

    /**
     * Put the Malayalam where the typed word is, in the field itself.
     *
     * A single space is left after it, so that the next word can be typed
     * straight away and the user does not have to press the space bar
     * between the words.
     */
    private fun replaceWord(word: String, trailingSpace: Boolean = true) {
        val connection = currentInputConnection ?: return
        val typed = currentWord()
        if (typed.isEmpty()) {
            connection.commitText(word, 1)
            if (trailingSpace) connection.commitText(" ", 1)
            return
        }
        connection.finishComposingText()
        connection.deleteSurroundingTextInCodePoints(typed.length, 0)
        connection.commitText(word, 1)
        if (trailingSpace) connection.commitText(" ", 1)
    }

    private fun switchKeyboard() {
        val manager = getSystemService(android.content.Context.INPUT_METHOD_SERVICE)
                as android.view.inputmethod.InputMethodManager
        manager.showInputMethodPicker()
    }

    // ------------------------------------------------------------ suggestions

    private fun firstSuggestion(): String? {
        if (!::strip.isInitialized) return null
        if (strip.childCount == 0) return null
        return (strip.getChildAt(0) as? TextView)?.text?.toString()
    }

    /**
     * Update the suggestion strip.
     *
     * The rules and the word list answer straight away. The model runs on a
     * background thread, because it is too slow for the main thread.
     */
    private fun refresh() {
        val word = currentWord()
        if (word.isEmpty()) {
            show(emptyList())
            return
        }
        show(keyboard.quick(word))
        worker?.interrupt()
        worker = Thread {
            val ranked = try {
                keyboard.suggest(word)
            } catch (error: Exception) {
                Log.e(TAG, "the model failed", error)
                emptyList()
            }
            handler.post { if (currentWord() == word) show(ranked) }
        }.also { it.start() }
    }

    private fun show(words: List<String>) {
        if (!::strip.isInitialized) return
        Log.i(TAG, "suggestions: ${words.joinToString(" | ")}")
        strip.removeAllViews()
        for ((index, word) in words.withIndex()) {
            val button = android.widget.Button(this)
            button.text = word
            button.textSize = 16f
            button.isAllCaps = false
            button.setPadding(dp(14), 0, dp(14), 0)
            button.setBackgroundColor(android.graphics.Color.TRANSPARENT)
            // The first suggestion is the one the space bar commits, so it
            // gets the accent colour.
            val colour = if (index == 0) palette.accent else palette.function
            val text = if (index == 0) palette.accentText else palette.functionText
            button.setTextColor(text)
            button.background = rounded(colour, if (index == 0) 18f else 10f)
            button.setOnClickListener {
                replaceWord(word)
                // The word is gone, so the strip has nothing to say until the
                // next one is typed.
                refresh()
            }
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, dp(36),
            )
            params.setMargins(dp(4), 0, dp(4), 0)
            button.layoutParams = params
            strip.addView(button)
        }
    }

    private fun rounded(colour: Int, radius: Float): android.graphics.drawable.GradientDrawable {
        val shape = android.graphics.drawable.GradientDrawable()
        shape.setColor(colour)
        shape.cornerRadius = radius
        return shape
    }

    // ------------------------------------------------------------------ theme

    private fun applyTheme() {
        if (!::view.isInitialized) return
        view.findViewById<View>(R.id.keyboard_root)?.setBackgroundColor(palette.background)
        view.findViewById<View>(R.id.strip_scroll)?.setBackgroundColor(palette.panel)
        view.findViewById<View>(R.id.strip_divider)?.setBackgroundColor(palette.function)
        strip.setBackgroundColor(palette.panel)
        keys.setBackgroundColor(palette.background)
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    companion object {
        const val TAG = "MangoKeys"
    }
}
