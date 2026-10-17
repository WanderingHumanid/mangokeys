package com.mangokeyboard.transliterator

import org.json.JSONArray
import org.json.JSONObject

/**
 * Character level tokenizer for the transliteration model.
 *
 * The vocabulary file is the same one the training run writes. It holds the
 * source characters and the target characters as two flat lists.
 */
class Tokenizer(private val payload: JSONObject) {

    private val srcIds: Map<String, Int> = HashMap<String, Int>().apply {
        val array = payload.getJSONArray("src")
        for (index in 0 until array.length()) {
            put(array.getString(index), index)
        }
    }

    private val tgtChars: JSONArray = payload.getJSONArray("tgt")

    val bosId: Int = tgtChars.indexOf("<bos>")
    val eosId: Int = tgtChars.indexOf("<eos>")
    val unkId: Int = tgtChars.indexOf("<unk>")

    fun encodeSource(text: String): IntArray {
        val ids = ArrayList<Int>(text.length)
        for (char in text) {
            ids.add(srcIds[char.toString()] ?: unkId)
        }
        return ids.toIntArray()
    }

    fun decodeTarget(ids: IntArray): String {
        val out = StringBuilder()
        for (id in ids) {
            if (id < 0 || id >= tgtChars.length()) continue
            val char = tgtChars.getString(id)
            if (char.startsWith("<")) continue
            out.append(char)
        }
        return out.toString()
    }
}
