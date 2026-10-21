package com.mangokeyboard.transliterator

import java.io.File

/**
 * The Malayalam character language model on the phone.
 *
 * It is trained on the same corpus as the transliterator and it ranks the
 * words that the word list and the model produce. The file is the one that
 * scripts/train_ngram.py writes.
 */
class CharNgram private constructor(
    val order: Int,
    private val alpha: Double,
    private val tables: Array<HashMap<String, HashMap<String, Int>>>,
    private val totals: Array<HashMap<String, Int>>,
    private val vocabulary: Int,
) {
    /** The probability of a letter given the letters before it. */
    fun probability(context: List<String>, symbol: String): Double {
        var length = minOf(context.size, order)
        while (length > 0) {
            val key = context.subList(context.size - length, context.size).joinToString(SEPARATOR)
            val table = tables[length - 1][key] ?: return backoff(symbol)
            val total = totals[length - 1][key] ?: 0
            if (total == 0) return backoff(symbol)
            val count = table[symbol] ?: 0
            return (count + alpha) / (total + alpha * vocabulary)
        }
        return backoff(symbol)
    }

    private fun backoff(symbol: String): Double {
        val table = tables[0][""]
        val total = totals[0][""] ?: 1
        val count = table?.get(symbol) ?: 0
        return (count + alpha) / (total + alpha * vocabulary)
    }

    /** The log probability of a word. Longer words score lower. */
    fun score(text: String): Double {
        val symbols = ArrayList<String>()
        symbols.add(START)
        symbols.addAll(letters(text))
        symbols.add(END)
        var total = 0.0
        for (index in symbols.indices) {
            total += Math.log(probability(symbols.subList(0, index), symbols[index]))
        }
        return total
    }

    /** The log probability per letter, so that words of different lengths fit. */
    fun meanScore(text: String): Double {
        if (text.isEmpty()) return 0.0
        return score(text) / (letters(text).size + 2)
    }

    companion object {
        private const val START = "<s>"
        private const val END = "</s>"
        private const val SEPARATOR = ""

        /** Read the model from a file. */
        fun load(path: File): CharNgram {
            val order: Int
            val alpha: Double
            val rows = ArrayList<Array<Any>>()
            val symbols = HashSet<String>()
            var readOrder = 3
            var readAlpha = 0.1
            path.readLines(Charsets.UTF_8).forEach { line ->
                if (line.isEmpty()) return@forEach
                val parts = line.split("\t")
                when (parts[0]) {
                    "order" -> readOrder = parts[1].toInt()
                    "alpha" -> readAlpha = parts[1].toDouble()
                    "ngram" -> {
                        val length = parts[1].toInt()
                        val context = parts.subList(2, parts.size - 2).joinToString(SEPARATOR)
                        val symbol = parts[parts.size - 2]
                        val count = parts[parts.size - 1].toInt()
                        rows.add(arrayOf(length, context, symbol, count))
                        symbols.add(symbol)
                    }
                }
            }
            order = readOrder
            alpha = readAlpha
            val tables = Array(order) { HashMap<String, HashMap<String, Int>>() }
            for (row in rows) {
                val length = row[0] as Int
                val context = row[1] as String
                val symbol = row[2] as String
                val count = row[3] as Int
                val table = tables[length - 1].getOrPut(context) { HashMap() }
                table[symbol] = (table[symbol] ?: 0) + count
            }
            val totals = Array(order) { HashMap<String, Int>() }
            for (index in 0 until order) {
                for ((context, counts) in tables[index]) {
                    totals[index][context] = counts.values.sum()
                }
            }
            return CharNgram(order, alpha, tables, totals, maxOf(symbols.size, 1))
        }
    }
}
