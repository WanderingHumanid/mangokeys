package com.mangokeyboard.transliterator

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

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
    onnxDirectory: java.io.File,
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
            setOptimizationLevel(OrtSession.SessionOptions.OrtOptimizationLevel.ORT_ENABLE_ALL)
        }
        encoder = environment.createSession(onnxDirectory.resolve("encoder.onnx"), options)
        decoder = environment.createSession(onnxDirectory.resolve("decoder.onnx"), options)
        tokenizer = Tokenizer(JSONObject(tokenizerJson))
    }

    /** Returns the top candidate Malayalam spelling for a Manglish word. */
    fun transliterate(roman: String): String {
        val candidates = candidates(roman, 1)
        return candidates.firstOrNull() ?: ""
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

        val memory = encode(src, srcMask)
        val width = minOf(beam, maxTgt)

        val beamMemory = memory.copyOf(memory.size * width)
        for (slot in 0 until width) {
            val offset = slot * memory.size
            for (index in memory.indices) {
                beamMemory[offset + index] = memory[index]
            }
        }
        val beamMask = BooleanArray(maxSrc * width)
        for (slot in 0 until width) {
            for (index in 0 until maxSrc) {
                beamMask[slot * maxSrc + index] = srcMask[index]
            }
        }

        var tokens = LongArray(width) { tokenizer.bosId.toLong() }
        var scores = DoubleArray(width) { Double.NEGATIVE_INFINITY }
        scores[0] = 0.0
        val finished = mutableListOf<Pair<Double, IntArray>>()

        for (step in 0 until maxTgt) {
            val logits = decode(tokens, beamMemory, beamMask)
            val vocab = logits.size / width

            val candidates = ArrayList<Pair<Double, Int>>(width)
            for (slot in 0 until width) {
                val row = logits.copyOfRange(slot * vocab, (slot + 1) * vocab)
                val logProbs = logSoftmax(row)
                for (token in 0 until vocab) {
                    candidates.add(Pair(scores[slot] + logProbs[token], slot * vocab + token))
                }
            }
            candidates.sortByDescending { it.first }

            val nextTokens = LongArray(width)
            val nextScores = DoubleArray(width)
            val nextPrefix = LongArray(width * (tokens.size + 1))
            for (slot in 0 until width) {
                val (score, index) = candidates[slot]
                val parent = index / vocab
                val token = index % vocab
                val prefixStart = slot * (tokens.size + 1)
                System.arraycopy(tokens, parent * tokens.size, nextPrefix, prefixStart, tokens.size)
                nextPrefix[prefixStart + tokens.size] = token.toLong()
                nextTokens[slot] = token.toLong()
                nextScores[slot] = score

                if (token == tokenizer.eosId) {
                    val sequence = nextPrefix.copyOfRange(prefixStart, prefixStart + tokens.size)
                    finished.add(Pair(score / maxOf(sequence.size, 1), sequence))
                }
            }

            tokens = nextPrefix
            scores = nextScores
            var alive = false
            for (slot in 0 until width) {
                if (nextTokens[slot] == tokenizer.eosId.toLong()) {
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
                val sequence = tokens.copyOfRange(slot * tokens.size, (slot + 1) * tokens.size)
                finished.add(Pair(scores[slot] / maxOf(sequence.size, 1), sequence))
            }
        }

        return finished.sortedByDescending { it.first }
            .take(nBest)
            .map { Normalizer.normalizeMalayalam(tokenizer.decodeTarget(it.second)) }
    }

    fun close() {
        encoder.close()
        decoder.close()
    }

    private fun encode(src: LongArray, srcMask: BooleanArray): FloatArray {
        val srcTensor = OnnxTensor.createTensor(environment, src)
        val maskTensor = OnnxTensor.createTensor(environment, srcMask)
        val result = try {
            encoder.run(mapOf("src_ids" to srcTensor, "src_mask" to maskTensor)).use { outputs ->
                outputs[0].value as FloatArray
            }
        } finally {
            srcTensor.close()
            maskTensor.close()
        }
        return result
    }

    private fun decode(tokens: LongArray, memory: FloatArray, srcMask: BooleanArray): FloatArray {
        val prefix = OnnxTensor.createTensor(environment, tokens)
        val memoryTensor = OnnxTensor.createTensor(environment, memory)
        val maskTensor = OnnxTensor.createTensor(environment, srcMask)
        val selfMask = OnnxTensor.createTensor(environment, causalMask(tokens.size))
        return try {
            decoder.run(
                mapOf(
                    "tgt_prefix" to prefix,
                    "memory" to memoryTensor,
                    "src_mask" to maskTensor,
                    "self_mask" to selfMask,
                )
            ).use { outputs ->
                outputs[0].value as FloatArray
            }
        } finally {
            prefix.close()
            memoryTensor.close()
            maskTensor.close()
            selfMask.close()
        }
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

    private fun logSoftmax(values: FloatArray): DoubleArray {
        var max = Double.NEGATIVE_INFINITY
        for (value in values) {
            if (value > max) max = value.toDouble()
        }
        var sum = 0.0
        val out = DoubleArray(values.size)
        for (index in values.indices) {
            out[index] = values[index] - max
            sum += Math.exp(out[index])
        }
        val logSum = Math.log(sum)
        for (index in out.indices) {
            out[index] -= logSum
        }
        return out
    }
}
