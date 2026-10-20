package com.mangokeyboard.transliterator

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Runs the exported transliteration model on device.
 *
 * The encoder and the decoder step are two separate graphs. The encoder runs
 * once per word and the decoder is then stepped until the end token comes out.
 * The causal mask is handed to the decoder as an input because a mask built
 * from a length that is only known at run time would be frozen at export time.
 *
 * The class holds no Android types so that it can be tested on a desktop.
 */
class Transliterator(
    onnxDirectory: File,
    tokenizerJson: String,
    private val maxSrc: Int = 48,
    private val maxTgt: Int = 64,
    private val beam: Int = 4,
) {
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession
    private val decoder: OrtSession
    private val tokenizer: Tokenizer

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        encoder = environment.createSession(onnxDirectory.resolve("encoder.onnx").path, options)
        decoder = environment.createSession(onnxDirectory.resolve("decoder.onnx").path, options)
        tokenizer = Tokenizer(JSONObject(tokenizerJson))
    }

    /** Transliterates a word or a whole phrase, one word at a time. */
    fun transliterate(roman: String): String {
        val words = Normalizer.normalizeRoman(roman).split(" ").filter { it.isNotEmpty() }
        if (words.isEmpty()) return ""
        if (words.size == 1) {
            val result = candidates(words[0], 1)
            return result.firstOrNull() ?: ""
        }
        val parts = words.map { word -> candidates(word, 1).firstOrNull() ?: "" }
        return parts.joinToString(" ")
    }

    /** Returns the top n Malayalam candidates for a Manglish word. */
    fun candidates(roman: String, nBest: Int = 5): List<String> {
        val clean = Normalizer.normalizeRoman(roman)
        if (clean.isEmpty()) return emptyList()

        val srcIds = tokenizer.encodeSource(clean)
        val src = LongArray(maxSrc)
        val srcMask = BooleanArray(maxSrc)
        for (index in srcIds.indices) {
            src[index] = srcIds[index].toLong()
            srcMask[index] = true
        }

        val encoded = encode(src, srcMask)
        val memory = encoded.first
        val memorySeq = encoded.second[1].toInt()
        val memoryDim = encoded.second[2].toInt()
        val width = minOf(beam, maxTgt)

        // Every beam slot works on the same encoder output.
        val memorySize = memorySeq * memoryDim
        val beamMemory = FloatArray(memorySize * width)
        for (slot in 0 until width) {
            System.arraycopy(memory, 0, beamMemory, slot * memorySize, memorySize)
        }
        val beamMask = BooleanArray(maxSrc * width)
        for (slot in 0 until width) {
            System.arraycopy(srcMask, 0, beamMask, slot * maxSrc, maxSrc)
        }

        var prefixLength = 1
        var tokens = LongArray(width)
        for (index in 0 until width) tokens[index] = tokenizer.bosId.toLong()

        var scores = DoubleArray(width) { Double.NEGATIVE_INFINITY }
        scores[0] = 0.0
        val finished = mutableListOf<Pair<Double, IntArray>>()
        var bestFinished = Double.NEGATIVE_INFINITY

        for (step in 0 until maxTgt) {
            val logits = decode(tokens, prefixLength, beamMemory, memorySeq, memoryDim, beamMask)
            val vocab = logits.size / width

            val ranked = ArrayList<Pair<Double, Int>>(width * vocab)
            for (slot in 0 until width) {
                val row = logSoftmax(logits, slot * vocab, vocab)
                for (token in 0 until vocab) {
                    ranked.add(Pair(scores[slot] + row[token], slot * vocab + token))
                }
            }
            ranked.sortByDescending { it.first }

            val nextLength = prefixLength + 1
            val nextTokens = LongArray(width)
            val nextScores = DoubleArray(width)
            val nextPrefix = LongArray(width * nextLength)
            for (slot in 0 until width) {
                val (score, index) = ranked[slot]
                val parent = index / vocab
                val token = index % vocab
                val prefixStart = slot * nextLength
                System.arraycopy(
                    tokens,
                    parent * prefixLength,
                    nextPrefix,
                    prefixStart,
                    prefixLength,
                )
                nextPrefix[prefixStart + prefixLength] = token.toLong()
                nextTokens[slot] = token.toLong()
                nextScores[slot] = score

                if (token == tokenizer.eosId) {
                    val sequence = nextPrefix.copyOfRange(prefixStart, prefixStart + prefixLength)
                    finished.add(Pair(score, sequence.map { it.toInt() }.toIntArray()))
                    if (score > bestFinished) bestFinished = score
                }
            }

            tokens = nextPrefix
            prefixLength = nextLength
            scores = nextScores
            var alive = false
            for (slot in 0 until width) {
                // A live beam can only lose score, so it is hopeless once it is
                // behind the best finished hypothesis.
                if (scores[slot] < bestFinished) {
                    scores[slot] = Double.NEGATIVE_INFINITY
                } else if (scores[slot] > Double.NEGATIVE_INFINITY) {
                    alive = true
                }
            }
            if (!alive) break
        }

        if (finished.isEmpty()) {
            for (slot in 0 until width) {
                if (scores[slot] == Double.NEGATIVE_INFINITY) continue
                val sequence = tokens.copyOfRange(slot * prefixLength, (slot + 1) * prefixLength)
                finished.add(Pair(scores[slot], sequence.map { it.toInt() }.toIntArray()))
            }
        }

        val seen = mutableSetOf<String>()
        val results = mutableListOf<String>()
        for (item in finished.sortedByDescending { it.first }) {
            val text = Normalizer.normalizeMalayalam(tokenizer.decodeTarget(item.second))
            if (!seen.add(text)) continue
            results.add(text)
            if (results.size >= nBest) break
        }
        return results
    }

    fun close() {
        encoder.close()
        decoder.close()
    }

    /** Runs the encoder and returns the memory and its shape. */
    private fun encode(src: LongArray, srcMask: BooleanArray): Pair<FloatArray, LongArray> {
        val srcTensor = OnnxTensor.createTensor(
            environment,
            LongBuffer.wrap(src),
            longArrayOf(1, src.size.toLong()),
        )
        val maskTensor = boolTensor(srcMask, longArrayOf(1, srcMask.size.toLong()))
        return try {
            encoder.run(mapOf("src_ids" to srcTensor, "src_mask" to maskTensor)).use { outputs ->
                val tensor = outputs[0] as OnnxTensor
                Pair(flatten(tensor.value), tensor.info.shape)
            }
        } finally {
            srcTensor.close()
            maskTensor.close()
        }
    }

    private fun decode(
        tokens: LongArray,
        prefixLength: Int,
        memory: FloatArray,
        memorySeq: Int,
        memoryDim: Int,
        srcMask: BooleanArray,
    ): FloatArray {
        val width = srcMask.size / maxSrc
        val prefix = OnnxTensor.createTensor(
            environment,
            LongBuffer.wrap(tokens),
            longArrayOf(width.toLong(), prefixLength.toLong()),
        )
        val memoryTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(memory),
            longArrayOf(width.toLong(), memorySeq.toLong(), memoryDim.toLong()),
        )
        val maskTensor = boolTensor(srcMask, longArrayOf(width.toLong(), maxSrc.toLong()))
        val selfMask = boolTensor(
            causalMask(prefixLength),
            longArrayOf(1, 1, prefixLength.toLong(), prefixLength.toLong()),
        )
        return try {
            decoder.run(
                mapOf(
                    "tgt_prefix" to prefix,
                    "memory" to memoryTensor,
                    "src_mask" to maskTensor,
                    "self_mask" to selfMask,
                )
            ).use { outputs ->
                flatten(outputs[0].value)
            }
        } finally {
            prefix.close()
            memoryTensor.close()
            maskTensor.close()
            selfMask.close()
        }
    }

    /**
     * The runtime hands a multi dimensional tensor back as nested arrays, and
     * the search loop wants one flat row per beam.
     */
    private fun flatten(value: Any): FloatArray {
        val out = ArrayList<Float>()
        collect(value, out)
        return out.toFloatArray()
    }

    private fun collect(value: Any, out: MutableList<Float>) {
        if (value is FloatArray) {
            for (item in value) out.add(item)
            return
        }
        for (item in value as Array<*>) {
            if (item != null) collect(item, out)
        }
    }

    /** A boolean tensor. The runtime has no boolean buffer, so it goes as bytes. */
    private fun boolTensor(data: BooleanArray, shape: LongArray): OnnxTensor {
        val bytes = ByteArray(data.size)
        for (index in data.indices) {
            bytes[index] = if (data[index]) 1 else 0
        }
        return OnnxTensor.createTensor(
            environment,
            ByteBuffer.wrap(bytes),
            shape,
            OnnxJavaType.BOOL,
        )
    }

    private fun boolTensor(data: ByteArray, shape: LongArray): OnnxTensor {
        return OnnxTensor.createTensor(
            environment,
            ByteBuffer.wrap(data),
            shape,
            OnnxJavaType.BOOL,
        )
    }

    private fun causalMask(length: Int): ByteArray {
        val buffer = ByteBuffer.allocate(length * length).order(ByteOrder.nativeOrder())
        for (row in 0 until length) {
            for (column in 0 until length) {
                buffer.put(if (column > row) 1 else 0)
            }
        }
        return buffer.array()
    }

    private fun logSoftmax(values: FloatArray, offset: Int, size: Int): DoubleArray {
        var max = Double.NEGATIVE_INFINITY
        for (index in 0 until size) {
            if (values[offset + index] > max) max = values[offset + index].toDouble()
        }
        var sum = 0.0
        val out = DoubleArray(size)
        for (index in 0 until size) {
            out[index] = values[offset + index] - max
            sum += Math.exp(out[index])
        }
        val logSum = Math.log(sum)
        for (index in 0 until size) out[index] -= logSum
        return out
    }
}
