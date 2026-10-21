/**
 * The on-screen keyboard of the demo.
 *
 * It is the keyboard of the Android app, drawn on the page: three rows of
 * letters, the shift on the left and the backspace on the right, and the
 * numbers, comma, space, period and enter on the last row. The rows and the
 * weights are the ones in the Kotlin file, so the demo and the app cannot
 * drift apart. The test that checks that is tests/test_web_layout.py.
 */

const LETTER_ROWS = [
  ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
  ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
  ["z", "x", "c", "v", "b", "n", "m"],
];

const NUMBER_ROWS = [
  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
  ["@", "#", "$", "%", "&", "*", "-", "+", "/", "("],
  [".", ",", "?", "!", "'", '"', ":", ";"],
];

/** The icons are drawn, so that they all carry the same stroke and weight. */
const ICONS = {
  shift: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20V5.5"/><path d="M6.5 11 12 5.5 17.5 11"/></svg>`,
  shiftLocked: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 18.5V7"/><path d="M7.5 11 12 6.5 16.5 11"/><path d="M8 21h8"/></svg>`,
  backspace: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5.5h10.5A1.5 1.5 0 0 1 21 7v10a1.5 1.5 0 0 1-1.5 1.5H9L3 12z"/><path d="M12.5 9.5l5 5"/><path d="M17.5 9.5l-5 5"/></svg>`,
  enter: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 10 5 14l4 4"/><path d="M5 14h11a4 4 0 0 0 4-4V5.5"/></svg>`,
};

const SHIFT_OFF = "OFF";
const SHIFT_ON = "ON";
const SHIFT_LOCKED = "LOCKED";

export class OnScreenKeyboard {
  /**
   * @param {HTMLElement} root the element the keys are built in
   * @param {object} handlers the callbacks for every kind of key
   */
  constructor(root, handlers) {
    this.root = root;
    this.handlers = handlers;
    this.shift = SHIFT_OFF;
    this.numbers = false;
    this.buttons = new Map();
    this.build();
  }

  build() {
    this.root.innerHTML = "";
    this.buttons.clear();
    const rows = this.numbers ? NUMBER_ROWS : LETTER_ROWS;

    this.root.appendChild(this.row(rows[0], "letter-row", 1));
    this.root.appendChild(this.row(rows[1], "letter-row inset", 1));

    const third = document.createElement("div");
    third.className = "key-row";
    if (this.numbers) {
      third.appendChild(this.key("abc", "function", "letters", 1.4));
    } else {
      third.appendChild(this.key("shift", "function", "shift", 1.4, this.shiftIcon()));
    }
    for (const label of rows[2]) {
      third.appendChild(this.key(label, "letter", "char", 1));
    }
    third.appendChild(this.key("backspace", "function", "backspace", 1.4, ICONS.backspace));
    this.root.appendChild(third);

    const last = document.createElement("div");
    last.className = "key-row";
    last.appendChild(
      this.key(this.numbers ? "abc" : "123", "function", "numbers", 1.2),
    );
    last.appendChild(this.key(",", "letter", "char", 1));
    last.appendChild(this.key("space", "letter", "space", 4.4));
    last.appendChild(this.key(".", "letter", "char", 1));
    last.appendChild(this.key("enter", "accent", "enter", 1.4, ICONS.enter));
    this.root.appendChild(last);
  }

  row(labels, className, weight) {
    const line = document.createElement("div");
    line.className = `key-row ${className}`;
    for (const label of labels) {
      line.appendChild(this.key(label, "letter", "char", weight));
    }
    return line;
  }

  key(label, kind, action, weight, icon = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `key key-${kind}`;
    button.dataset.action = action;
    button.dataset.key = label;
    button.style.flexGrow = String(weight);
    if (icon) {
      button.classList.add("key-icon");
      button.innerHTML = icon;
    } else {
      button.textContent = label;
    }
    button.addEventListener("click", (event) => {
      event.preventDefault();
      this.press(label, action);
    });
    this.buttons.set(label, button);
    return button;
  }

  press(label, action) {
    if (action === "char") this.handlers.onCharacter(label);
    else if (action === "space") this.handlers.onSpace();
    else if (action === "backspace") this.handlers.onBackspace();
    else if (action === "enter") this.handlers.onEnter();
    else if (action === "shift") this.cycleShift();
    else if (action === "numbers") this.setNumbers(true);
    else if (action === "letters") this.setNumbers(false);
  }

  shiftIcon() {
    return this.shift === SHIFT_LOCKED ? ICONS.shiftLocked : ICONS.shift;
  }

  cycleShift() {
    if (this.shift === SHIFT_OFF) this.shift = SHIFT_ON;
    else if (this.shift === SHIFT_ON) this.shift = SHIFT_LOCKED;
    else this.shift = SHIFT_OFF;
    this.build();
  }

  /** The letter that the next character gets, given the state of the shift. */
  currentLetter(letter) {
    if (this.shift === SHIFT_ON) return letter.toUpperCase();
    return letter;
  }

  /** The shift goes back to off after one letter, unless it is locked. */
  consumeShift() {
    if (this.shift === SHIFT_ON) {
      this.shift = SHIFT_OFF;
      this.build();
    }
  }

  setNumbers(value) {
    this.numbers = value;
    this.build();
  }

  /** Light the key that the physical keyboard just pressed. */
  flash(label) {
    const button = this.buttons.get(label);
    if (!button) return;
    button.classList.add("pressed");
    window.setTimeout(() => button.classList.remove("pressed"), 130);
  }

  /**
   * The space bar carries the accent when there is a candidate to commit,
   * because in that moment the space bar does something.
   */
  setSpaceActive(active) {
    const button = this.buttons.get("space");
    if (!button) return;
    button.classList.toggle("key-ready", active);
  }
}

export { ICONS, LETTER_ROWS, NUMBER_ROWS };
