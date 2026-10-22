package com.mangokeyboard

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.util.TypedValue
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.View
import android.widget.Button
import android.widget.LinearLayout

/**
 * The keys of the keyboard.
 *
 * The layout is the standard one: three rows of letters, the shift on the
 * left and the backspace on the right, and a last row with the numbers, the
 * space and the enter. The numbers are a second layer that the 123 key
 * switches to.
 *
 * The rows are built in code rather than in a layout file, because a layout
 * file for a keyboard is a thousand lines of almost the same thing.
 */
object KeyboardLayout {

    val LETTERS = listOf(
        listOf("q", "w", "e", "r", "t", "y", "u", "i", "o", "p"),
        listOf("a", "s", "d", "f", "g", "h", "j", "k", "l"),
        listOf("z", "x", "c", "v", "b", "n", "m"),
    )

    val NUMBERS = listOf(
        listOf("1", "2", "3", "4", "5", "6", "7", "8", "9", "0"),
        listOf("@", "#", "$", "%", "&", "*", "-", "+", "/", "("),
        listOf(".", ",", "?", "!", "'", "\"", ":", ";"),
    )

    /** The state of the shift key. */
    enum class Shift { OFF, ON, LOCKED }

    /** Whether the keys vibrate when they are pressed. */
    private var haptics = true

    fun build(
        context: Context,
        container: LinearLayout,
        palette: Palette,
        shift: Shift,
        numbers: Boolean,
        onKey: (String) -> Unit,
        onSpace: () -> Unit,
        onShift: () -> Unit,
        onNumbers: () -> Unit,
        onLetters: () -> Unit,
        onBackspace: () -> Unit,
        onEnter: () -> Unit,
        onGlobe: () -> Unit,
        haptics: Boolean = true,
    ) {
        this.haptics = haptics
        container.removeAllViews()
        container.setBackgroundColor(palette.background)
        val rows = if (numbers) NUMBERS else LETTERS

        // The letters are shown in capitals while the shift is on, so that
        // the keyboard says what it is going to type.
        val shifted = shift == Shift.ON || shift == Shift.LOCKED

        container.addView(letterRow(context, rows[0], palette, 0, shifted, onKey))
        container.addView(letterRow(context, rows[1], palette, dp(context, 12), shifted, onKey))

        val third = LinearLayout(context)
        third.orientation = LinearLayout.HORIZONTAL
        third.layoutParams = rowParams()
        if (numbers) {
            third.addView(
                key(context, "abc", 1.4f, palette.function, palette.functionText) { onLetters() },
            )
        } else {
            // The shift key is lit while it is on, the way it is on the
            // browser demo.
            val colour = if (shifted) palette.accent else palette.function
            val text = if (shifted) palette.accentText else palette.functionText
            third.addView(
                key(context, shiftLabel(shift), 1.4f, colour, text) {
                    onShift()
                },
            )
        }
        for (label in rows[2]) {
            third.addView(
                key(context, shownLabel(label, shifted), 1f, palette.key, palette.keyText) {
                    onKey(label)
                },
            )
        }
        third.addView(
            key(context, "⌫", 1.4f, palette.function, palette.functionText) { onBackspace() },
        )
        container.addView(third)

        val last = LinearLayout(context)
        last.orientation = LinearLayout.HORIZONTAL
        last.layoutParams = rowParams()
        last.addView(
            key(context, if (numbers) "abc" else "123", 1.2f, palette.function, palette.functionText) {
                if (numbers) onLetters() else onNumbers()
            },
        )
        last.addView(
            key(context, ",", 1f, palette.key, palette.keyText) { onKey(",") },
        )
        last.addView(
            key(context, "space", 4.4f, palette.key, palette.keyText) { onSpace() },
        )
        last.addView(
            key(context, ".", 1f, palette.key, palette.keyText) { onKey(".") },
        )
        last.addView(
            key(context, "↵", 1.4f, palette.accent, palette.accentText) { onEnter() },
        )
        container.addView(last)
    }

    fun shiftLabel(shift: Shift): String = when (shift) {
        Shift.OFF -> "⇧"
        Shift.ON -> "⇧"
        Shift.LOCKED -> "⇪"
    }

    private fun letterRow(
        context: Context,
        row: List<String>,
        palette: Palette,
        inset: Int,
        shifted: Boolean,
        onKey: (String) -> Unit,
    ): View {
        val line = LinearLayout(context)
        line.orientation = LinearLayout.HORIZONTAL
        line.layoutParams = rowParams()
        if (inset > 0) {
            line.setPadding(inset, 0, inset, 0)
        }
        for (label in row) {
            line.addView(
                key(context, shownLabel(label, shifted), 1f, palette.key, palette.keyText) {
                    onKey(label)
                },
            )
        }
        return line
    }

    /**
     * The label of a key.
     *
     * Only the letters change. The digits, the symbols and the words stay as
     * they are.
     */
    private fun shownLabel(label: String, shifted: Boolean): String {
        if (!shifted) return label
        if (label.length != 1 || !label[0].isLowerCase()) return label
        return label.uppercase()
    }

    private fun rowParams(): LinearLayout.LayoutParams = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.WRAP_CONTENT,
    )

    private fun key(
        context: Context,
        label: String,
        weight: Float,
        color: Int,
        textColor: Int,
        onClick: () -> Unit,
    ): View {
        val button = Button(context)
        button.text = label
        button.textSize = 17f
        button.setTextColor(textColor)
        button.isAllCaps = false
        button.gravity = Gravity.CENTER
        button.setPadding(0, 0, 0, 0)
        button.setBackgroundColor(Color.TRANSPARENT)
        button.background = rounded(color)
        button.setOnClickListener {
            // The key feels the press. The feedback of a view does not need a
            // permission, so the app still asks for none.
            if (haptics) {
                button.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
            }
            onClick()
        }

        val params = LinearLayout.LayoutParams(0, dp(context, 44), weight)
        params.setMargins(dp(context, 2), dp(context, 2), dp(context, 2), dp(context, 2))
        button.layoutParams = params
        return button
    }

    private fun rounded(color: Int): GradientDrawable {
        val shape = GradientDrawable()
        shape.setColor(color)
        shape.cornerRadius = 10f
        return shape
    }

    private fun dp(context: Context, value: Int): Int {
        val metrics = context.resources.displayMetrics
        return TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            value.toFloat(),
            metrics,
        ).toInt()
    }
}
