package com.mangokeyboard

import android.content.Context
import java.io.File

/**
 * The model files of the keyboard.
 *
 * The onnx runtime needs file paths, so the assets are copied into the
 * private directory of the application the first time it starts. The service
 * and the test screen of the application both use them.
 */
class ModelAssets(private val context: Context) {

    val directory: File by lazy {
        val target = File(context.filesDir, "model")
        target.mkdirs()
        copy("encoder.onnx", File(target, "encoder.onnx"))
        copy("decoder.onnx", File(target, "decoder.onnx"))
        copy("tokenizer.json", File(target, "tokenizer.json"))
        val prefix = File(target, "prefix")
        prefix.mkdirs()
        copy("prefix-encoder.onnx", File(prefix, "encoder.onnx"))
        copy("prefix-decoder.onnx", File(prefix, "decoder.onnx"))
        val lexicon = File(target, "lexicon")
        lexicon.mkdirs()
        copy("lexicon/words.txt", File(lexicon, "words.txt"))
        copy("lexicon/index.bin", File(lexicon, "index.bin"))
        copy("ngram.txt", File(target, "ngram.txt"))
        target
    }

    private fun copy(name: String, target: File) {
        if (target.exists() && target.length() > 0) return
        try {
            context.assets.open(name).use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
        } catch (error: Exception) {
            // A missing file is not fatal. The keyboard falls back to the rules.
        }
    }
}
