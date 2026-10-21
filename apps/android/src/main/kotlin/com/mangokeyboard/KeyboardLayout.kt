package com.mangokeyboard

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout

/**
 * The keys of the keyboard.
 *
 * The rows are built in code rather than in a layout file, because a layout
 * file for a keyboard is a thousand lines of almost the same thing.
 */
object KeyboardLayout {
    val ROWS = listOf(
        listOf("q", "w", "e", "r", "t", "y", "u", "i", "o", "p"),
        listOf("a", "s", "d", "f", "g", "h", "j", "k", "l"),
        listOf("z", "x", "c", "v", "b", "n", "m"),
    )

    fun build(
        context: Context,
        container: LinearLayout,
        onKey: (String) -> Unit,
        onSpace: () -> Unit,
        onBackspace: () -> Unit,
        onEnter: () -> Unit,
        onGlobe: () -> Unit,
    ) {
        container.removeAllViews()
        for (row in ROWS) {
            container.addView(rowView(context, row, onKey))
        }
        container.addView(bottomRow(context, onSpace, onBackspace, onEnter, onGlobe))
    }

    private fun rowView(context: Context, row: List<String>, onKey: (String) -> Unit): View {
        val line = LinearLayout(context)
        line.orientation = LinearLayout.HORIZONTAL
        line.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT,
        )
        for (label in row) {
            line.addView(
                key(context, label, 1f) {
                    onKey(label)
                },
            )
        }
        return line
    }

    private fun bottomRow(
        context: Context,
        onSpace: () -> Unit,
        onBackspace: () -> Unit,
        onEnter: () -> Unit,
        onGlobe: () -> Unit,
    ): View {
        val line = LinearLayout(context)
        line.orientation = LinearLayout.HORIZONTAL
        line.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT,
        )
        line.addView(key(context, "globe", 1f) { onGlobe() })
        line.addView(key(context, "⌫", 1f) { onBackspace() })
        line.addView(key(context, "space", 4f) { onSpace() })
        line.addView(key(context, "↵", 1.4f) { onEnter() })
        return line
    }

    private fun key(context: Context, label: String, weight: Float, onClick: () -> Unit): View {
        val button = Button(context)
        button.text = label
        button.textSize = 16f
        button.setTextColor(Color.parseColor("#1A1C1E"))
        button.isAllCaps = false
        button.setOnClickListener { onClick() }

        val background = GradientDrawable()
        background.setColor(Color.parseColor("#FFFFFF"))
        background.cornerRadius = 12f
        background.setStroke(1, Color.parseColor("#D5D8DD"))
        button.background = background

        val params = LinearLayout.LayoutParams(0, dp(context, 46), weight)
        params.setMargins(dp(context, 2), dp(context, 2), dp(context, 2), dp(context, 2))
        button.layoutParams = params
        return button
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
