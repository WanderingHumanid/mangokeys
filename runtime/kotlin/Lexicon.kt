package com.mangokeyboard.transliterator

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.File

/**
 * The Malayalam word list on the phone.
 *
 * The three lookups are the ones the browser and the python runtime have:
 * the word closest to a string, the words that start with a prefix, and the
 * combination of the two that the keyboard uses. The files are written by
 * scripts/build_lexicon.py, so every runtime uses the same word list.
 *
 * The class holds no Android types, so that it can be tested on a desktop.
 */
class Lexicon(
    val words: Array<String> = emptyArray(),
    private val counts: IntArray = IntArray(0),
    private val indexKeys: LongArray = LongArray(0),
    private val indexIds: IntArray = IntArray(0),
    private val distance: Int = 2,
) {
    private val wordIds = HashMap<String, Int>(words.size * 2)
    private val skeletons = Array(words.size) { skeleton(words[it]) }
    private val buckets = HashMap<Int, HashMap<String, IntArray>>()

    init {
        for (id in words.indices) wordIds[words[id]] = id
        for (depth in 1..3) {
            val map = HashMap<String, MutableList<Int>>()
            for (id in words.indices) {
                val key = skeletons[id]
                if (key.length < depth) continue
                val name = key.substring(0, depth)
                map.getOrPut(name) { ArrayList() }.add(id)
            }
            val sorted = HashMap<String, IntArray>(map.size * 2)
            for ((name, ids) in map) {
                sorted[name] = ids.sortedByDescending { counts[it] }.toIntArray()
            }
            buckets[depth] = sorted
        }
    }

    /** Read the word list and its index from a directory. */
    fun load(directory: File, maxDistance: Int = 2): Lexicon {
        val words = ArrayList<String>()
        val counts = ArrayList<Int>()
        File(directory, "words.txt").readLines(Charsets.UTF_8).forEach { line ->
            if (line.isEmpty()) return@forEach
            val parts = line.split("\t")
            words.add(parts[0])
            counts.add(if (parts.size > 1) parts[1].toIntOrNull() ?: 0 else 0)
        }
        return build(words, counts, File(directory, "index.bin").readBytes(), maxDistance)
    }

    /** Read the word list and its index from memory. */
    fun load(wordsText: String, indexBytes: ByteArray, maxDistance: Int = 2): Lexicon {
        val words = ArrayList<String>()
        val counts = ArrayList<Int>()
        for (line in wordsText.split("\n")) {
            if (line.isEmpty()) continue
            val parts = line.split("\t")
            words.add(parts[0])
            counts.add(if (parts.size > 1) parts[1].toIntOrNull() ?: 0 else 0)
        }
        return build(words, counts, indexBytes, maxDistance)
    }

    private fun build(
        words: List<String>,
        counts: List<Int>,
        indexBytes: ByteArray,
        maxDistance: Int,
    ): Lexicon {
        val input = ByteBuffer.wrap(indexBytes).order(ByteOrder.LITTLE_ENDIAN)
        val wordCount = input.int
        val size = input.int
        val keys = LongArray(size)
        val ids = IntArray(size)
        for (position in 0 until size) {
            keys[position] = input.long
            ids[position] = input.int
        }
        if (wordCount != words.size) {
            throw IllegalStateException("the word list and the index do not match")
        }
        return Lexicon(words.toTypedArray(), counts.toIntArray(), keys, ids, maxDistance)
    }

    fun contains(word: String): Boolean = wordIds.containsKey(word)

    fun count(word: String): Int = wordIds[word]?.let { counts[it] } ?: 0

    /** The real words closest to a string, the best match first. */
    fun nearest(word: String, limit: Int = 8): List<String> {
        return nearestWithDistance(word, limit).map { it.first }
    }

    /** The real words closest to a string, with the distance to them. */
    fun nearestWithDistance(word: String, limit: Int = 8): List<Pair<String, Int>> {
        if (word.isEmpty()) return emptyList()
        val items = letters(word)
        val found = HashMap<Int, Int>()
        for (variant in deletions(items, distance)) {
            for (id in lookup(fnv1a(variant))) {
                val gap = levenshtein(items, letters(words[id]))
                if (gap > distance) continue
                val known = found[id]
                if (known == null || gap < known) found[id] = gap
            }
        }
        return found.entries
            .sortedWith(compareBy({ it.value }, { -counts[it.key] }, { words[it.key] }))
            .take(limit)
            .map { Pair(words[it.key], it.value) }
    }

    /** The word ids that the index holds under a key. */
    private fun lookup(key: Long): IntArray {
        var low = 0
        var high = indexKeys.size - 1
        while (low <= high) {
            val middle = (low + high) / 2
            val value = indexKeys[middle]
            if (value < key) {
                low = middle + 1
            } else if (value > key) {
                high = middle - 1
            } else {
                var start = middle
                while (start > 0 && indexKeys[start - 1] == key) start--
                var end = middle
                while (end + 1 < indexKeys.size && indexKeys[end + 1] == key) end++
                return indexIds.copyOfRange(start, end + 1)
            }
        }
        return IntArray(0)
    }

    /** The real words that start with a native prefix, the commonest first. */
    fun complete(prefix: String, limit: Int = 8): List<String> {
        if (prefix.isEmpty()) return emptyList()
        val key = skeleton(prefix)
        if (key.isEmpty()) return emptyList()
        val depth = minOf(key.length, 3)
        val bucket = buckets[depth]?.get(key.substring(0, depth)) ?: return emptyList()
        val out = ArrayList<String>()
        for (id in bucket) {
            if (!skeletons[id].startsWith(key)) continue
            out.add(words[id])
            if (out.size >= limit) break
        }
        return out
    }

    /** The real words around a string, with how far they are from it. */
    fun expand(text: String, limit: Int = 8): List<Pair<String, Int>> {
        if (text.isEmpty()) return emptyList()
        val out = ArrayList<Pair<String, Int>>()
        val seen = HashSet<String>()
        if (contains(text)) {
            out.add(Pair(text, 0))
            seen.add(text)
        }
        for (word in complete(text, limit * 2)) {
            if (!seen.add(word)) continue
            out.add(Pair(word, 1))
            if (out.size >= limit) return out
        }
        for (item in nearestWithDistance(text, limit * 2)) {
            if (!seen.add(item.first)) continue
            out.add(item)
            if (out.size >= limit) break
        }
        return out.take(limit)
    }

    val size: Int get() = words.size

    /** How far the lookup looks from a word. */
    val maxDistance: Int get() = distance
}

/** The consonant letters of a word, without the marks and the chillus. */
fun skeleton(word: String): String {
    val out = StringBuilder()
    for (char in word) {
        val letter: String = CHILLU_BASE[char] ?: char.toString()
        if (letter.length != 1) continue
        if (MARKS.contains(letter[0])) continue
        val code: Int = letter[0].code
        if (code in 0x0D15..0x0D39) out.append(letter)
    }
    return out.toString()
}

/** Split text into Malayalam letters, the same way the python side does. */
fun letters(text: String): List<String> {
    val out = ArrayList<String>()
    for (char in text) {
        val code = char.code
        val hanging = (code in 0x0D3B..0x0D4D) ||
            code == 0x0D01 || code == 0x0D02 || code == 0x0D03 ||
            code == 0x0D55 || code == 0x0D56 || code == 0x0D57 ||
            code == 0x200C || code == 0x200D
        if (hanging && out.isNotEmpty()) {
            out[out.size - 1] = out[out.size - 1] + char
        } else {
            out.add(char.toString())
        }
    }
    return out
}

/** The edit distance between two lists of letters. */
fun levenshtein(a: List<String>, b: List<String>): Int {
    val first = if (a.size < b.size) b else a
    val second = if (a.size < b.size) a else b
    var previous = IntArray(second.size + 1) { it }
    for (i in 1..first.size) {
        val current = IntArray(second.size + 1)
        current[0] = i
        for (j in 1..second.size) {
            val cost = if (first[i - 1] == second[j - 1]) 0 else 1
            current[j] = minOf(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        }
        previous = current
    }
    return previous[second.size]
}

/** Every string that comes from deleting up to `distance` items. */
private fun deletions(sequence: List<String>, distance: Int): Set<String> {
    val found = HashSet<String>()
    found.add(sequence.joinToString(SEPARATOR))
    var frontier = listOf(sequence)
    repeat(distance) {
        val next = ArrayList<List<String>>()
        for (item in frontier) {
            for (index in item.indices) {
                val shorter = item.toMutableList().also { it.removeAt(index) }
                next.add(shorter)
                found.add(shorter.joinToString(SEPARATOR))
            }
        }
        frontier = next
    }
    return found
}

/** The same 64 bit hash the python side uses to build the index. */
fun fnv1a(text: String): Long {
    var value = -3750763034362895579L
    for (byte in text.toByteArray(Charsets.UTF_8)) {
        value = value xor (byte.toLong() and 0xff)
        value *= 0x100000001b3L
    }
    return value
}

private const val SEPARATOR = "\u001F"

private val MARKS = "ാിീുൂൃൄ൅െേൈൊോൌൗ്ം".toSet()

private val CHILLU_BASE: Map<Char, String> = mapOf(
    'ൺ' to "ണ",
    'ൻ' to "ന",
    'ർ' to "ര",
    'ൽ' to "ല",
    'ൾ' to "ള",
    'ൿ' to "ക",
)
