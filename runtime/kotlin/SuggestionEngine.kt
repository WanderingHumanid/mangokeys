package com.mangokeyboard.transliterator

/**
 * The ranker of the keyboard on the phone.
 *
 * The rules and the model both propose words, the word list turns them into
 * real words, and the language model and the corpus frequency decide which
 * one comes first. The same code runs in the browser.
 */
class SuggestionEngine(
    private val lexicon: Lexicon,
    private val ngram: CharNgram?,
    private val weights: Weights = Weights(),
) {
    /** One word the keyboard is considering. */
    data class Candidate(
        val text: String,
        val source: String,
        var modelScore: Double = 0.0,
        var lmScore: Double = 0.0,
        var count: Int = 0,
        var known: Boolean = false,
        var completion: Boolean = false,
        var variation: Boolean = false,
        var total: Double = 0.0,
    )

    data class Weights(
        val model: Double = 1.0,
        val known: Double = 4.0,
        val completion: Double = 0.0,
        val variation: Double = -2.0,
        val language: Double = 0.0,
        val frequency: Double = 0.0,
        val rulePenalty: Double = -12.0,
        val limit: Int = 8,
    )

    /**
     * Rank the words that a Manglish string could mean.
     *
     * The model candidates come with the score the model gave them, so that
     * the ranking can prefer the answer the model is most sure about.
     */
    fun suggest(
        roman: String,
        modelCandidates: List<String> = emptyList(),
        modelScores: List<Double> = emptyList(),
        completionCandidates: List<String> = emptyList(),
        useRules: Boolean = true,
    ): List<Candidate> {
        val text = roman.trim()
        if (text.isEmpty()) return emptyList()

        val pool = LinkedHashMap<String, Candidate>()
        val best = if (modelScores.isEmpty()) 0.0 else modelScores[0]
        modelCandidates.forEachIndexed { index, word ->
            if (word.isEmpty()) return@forEachIndexed
            val score = if (index < modelScores.size) modelScores[index] - best else 0.0
            offer(pool, word, "model", score)
        }
        for (word in completionCandidates) {
            if (word.isEmpty()) continue
            add(pool, word, "completion", 0.0).completion = true
        }
        if (useRules) {
            for (word in RuleEngine().convert(text).take(4)) {
                offer(pool, word, "rules", weights.rulePenalty)
            }
        }

        for (candidate in pool.values) {
            candidate.count = lexicon.count(candidate.text)
            candidate.lmScore = ngram?.meanScore(candidate.text) ?: 0.0
            candidate.total = weights.model * candidate.modelScore +
                weights.known * (if (candidate.known) 1.0 else 0.0) +
                weights.completion * (if (candidate.completion) 1.0 else 0.0) +
                weights.variation * (if (candidate.variation) 1.0 else 0.0) +
                weights.language * candidate.lmScore +
                weights.frequency * Math.log1p(candidate.count.toDouble())
        }

        return pool.values.sortedByDescending { it.total }.take(weights.limit)
    }

    private fun offer(
        pool: LinkedHashMap<String, Candidate>,
        text: String,
        source: String,
        score: Double,
    ) {
        val candidate = add(pool, text, source, score)
        // The answer is a real word when the word list knows it. The words
        // that the model itself proposed get the bonus, because the model is
        // the better judge of what the user meant.
        if (source == "model" && lexicon.contains(text)) candidate.known = true

        // The words that start with the answer are the completion. They are
        // the answer only when the model did not give a whole word.
        if (!candidate.known) {
            for (word in lexicon.complete(text, 3)) {
                if (word == text) continue
                add(pool, word, source, score).completion = true
            }
        }

        // The words that resemble the answer are the variations.
        for (item in lexicon.nearestWithDistance(text, 3)) {
            if (item.first == text) continue
            add(pool, item.first, source, score).variation = true
        }
    }

    private fun add(
        pool: LinkedHashMap<String, Candidate>,
        text: String,
        source: String,
        score: Double,
    ): Candidate {
        val known = pool[text]
        if (known != null) {
            known.modelScore = maxOf(known.modelScore, score)
            return known
        }
        val candidate = Candidate(text, source, modelScore = score)
        pool[text] = candidate
        return candidate
    }
}
