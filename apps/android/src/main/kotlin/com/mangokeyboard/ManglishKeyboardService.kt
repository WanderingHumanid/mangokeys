package com.mangokeyboard

import android.inputmethodservice.InputMethodService
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.LinearLayout
import android.widget.TextView
import com.mangokeyboard.transliterator.CharNgram
import com.mangokeyboard.transliterator.Keyboard
import com.mangokeyboard.transliterator.Lexicon
import java.io.File

/**
 * The Manglish keyboard.
 *
 * The user types in Latin characters. On every keystroke the rules and the
 * word list give an answer straight away, and the model gives a better one
 * a moment later on a background thread. The suggestion strip shows the
 * words that exist, ranked by the language model.
 *
 * The model is loaded once, when the service is created. The onnx runtime
 * needs file paths, so the assets are copied into the private directory of
 * the application first.
 */
class ManglishKeyboardService : InputMethodService() {

    private lateinit var keyboard: Keyboard
    private lateinit var strip: LinearLayout
    private lateinit var composingView: TextView
    private val composing = StringBuilder()
    private val handler = Handler(Looper.getMainLooper())
    private var worker: Thread? = null
    private var inputView: View? = null

    override fun onCreate() {
        super.onCreate()
        val models = copyAssets()
        val lexicon = Lexicon().load(File(models, "lexicon"))
        val ngramFile = File(models, "ngram.txt")
        val ngram = if (ngramFile.exists()) CharNgram.load(ngramFile) else null
        val prefix = File(models, "prefix")
        keyboard = Keyboard(
            models,
            File(models, "tokenizer.json").readText(),
            lexicon,
            ngram,
            prefixModel = if (prefix.exists()) prefix else null,
        )
        Log.i(TAG, "the model is ready")
    }

    override fun onCreateInputView(): View {
        val view = layoutInflater.inflate(R.layout.keyboard, null)
        inputView = view
        strip = view.findViewById(R.id.suggestion_strip)
        composingView = view.findViewById(R.id.composing)

        KeyboardLayout.build(
            this,
            view.findViewById(R.id.keys),
            onKey = { type(it) },
            onSpace = { commitTop() },
            onBackspace = { backspace() },
            onEnter = { commitTop() },
            onGlobe = { switchKeyboard() },
        )
        return view
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        composing.clear()
        composingView.text = ""
        refresh()
    }

    override fun onFinishInput() {
        super.onFinishInput()
        composing.clear()
    }

    // ------------------------------------------------------------------ input

    private fun type(letter: String) {
        composing.append(letter)
        updateComposing()
        refresh()
    }

    private fun backspace() {
        if (composing.isNotEmpty()) composing.deleteCharAt(composing.length - 1)
        updateComposing()
        refresh()
    }

    private fun updateComposing() {
        composingView.text = composing.toString()
    }

    private fun commit(text: String) {
        val editor = currentInputConnection
        if (editor != null && text.isNotEmpty()) editor.commitText(text, 1)
        composing.clear()
        updateComposing()
        refresh()
    }

    private fun commitTop() {
        val top = firstSuggestion()
        if (top.isNullOrEmpty()) {
            commit(composing.toString())
            return
        }
        commit(top)
    }

    private fun switchKeyboard() {
        val manager = getSystemService(android.content.Context.INPUT_METHOD_SERVICE)
                as android.view.inputmethod.InputMethodManager
        manager.showInputMethodPicker()
    }

    // ------------------------------------------------------------ suggestions

    private fun firstSuggestion(): String? {
        val view = inputView ?: return null
        val strip = view.findViewById<LinearLayout>(R.id.suggestion_strip)
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
        val text = composing.toString()
        if (text.isEmpty()) {
            show(emptyList())
            return
        }
        show(keyboard.quick(text))
        worker?.interrupt()
        worker = Thread {
            val ranked = try {
                keyboard.suggest(text)
            } catch (error: Exception) {
                Log.e(TAG, "the model failed", error)
                emptyList()
            }
            handler.post { if (composing.toString() == text) show(ranked) }
        }.also { it.start() }
    }

    private fun show(words: List<String>) {
        Log.i(TAG, "suggestions: ${words.joinToString(" | ")}")
        strip.removeAllViews()
        for (word in words) {
            val button = android.widget.Button(this)
            button.text = word
            button.textSize = 15f
            button.isAllCaps = false
            button.setBackgroundColor(android.graphics.Color.WHITE)
            button.setOnClickListener { commit(word) }
            strip.addView(button)
        }
    }

    // ----------------------------------------------------------------- assets

    private fun copyAssets(): File {
        val target = File(filesDir, "model")
        target.mkdirs()
        for (name in listOf("encoder.onnx", "decoder.onnx", "tokenizer.json")) {
            copyAsset(name, File(target, name))
        }
        val prefix = File(target, "prefix")
        prefix.mkdirs()
        for (name in listOf("prefix-encoder.onnx", "prefix-decoder.onnx")) {
            copyAsset(name, File(prefix, name.removePrefix("prefix-")))
        }
        val lexicon = File(target, "lexicon")
        lexicon.mkdirs()
        for (name in listOf("words.txt", "index.bin")) {
            copyAsset("lexicon/$name", File(lexicon, name))
        }
        copyAsset("ngram.txt", File(target, "ngram.txt"))
        return target
    }

    private fun copyAsset(name: String, target: File) {
        if (target.exists() && target.length() > 0) return
        assets.open(name).use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        }
    }

    companion object {
        const val TAG = "ManglishKeyboard"
    }
}
