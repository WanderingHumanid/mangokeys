package com.mangokeyboard.transliterator

import java.io.File

/**
 * The keyboard on the phone.
 *
 * The model says which word the user probably meant, the word list says
 * which words exist, and the language model says which of them is the
 * natural one. There are two paths.
 *
 * `quick` uses the rules and the word list only. It takes well under a
 * millisecond, so it can run on every keystroke.
 *
 * `suggest` adds the model. It takes a few milliseconds, so it runs when the
 * typing pauses or when the word is committed.
 */
class Keyboard(
    onnxDirectory: File,
    tokenizerJson: String,
    private val lexicon: Lexicon,
    private val ngram: CharNgram? = null,
    beam: Int = 4,
    private val limit: Int = 8,
) {
    private val transliterator = Transliterator(
        onnxDirectory, tokenizerJson, maxSrc = 48, maxTgt = 64, beam = beam,
    )
    private val engine = SuggestionEngine(lexicon, ngram)

    /** The Malayalam of a word, without the word list. */
    fun transliterate(roman: String): String = transliterator.transliterate(roman)

    /** The suggestions from the rules and the word list alone. */
    fun quick(roman: String, nBest: Int = limit): List<String> {
        return engine.suggest(roman, useRules = true).take(nBest).map { it.text }
    }

    /** The suggestions with the model in the loop. */
    fun suggest(roman: String, nBest: Int = limit): List<String> {
        val scored = transliterator.scoredCandidates(roman, 5)
        val ranked = engine.suggest(
            roman,
            scored.map { it.first },
            scored.map { it.second },
        )
        return ranked.take(nBest).map { it.text }
    }
}
