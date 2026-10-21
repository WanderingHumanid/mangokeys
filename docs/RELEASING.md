# Releasing

This is the procedure for putting a version of MangoKeys on the GitHub
release page. The first release is `v0.01`.

The work is split in two. The GitHub Actions workflow builds and publishes,
and this file explains what a human has to do.

## What happens on a tag push

The `release.yml` workflow runs when a tag that starts with `v` is pushed. It

1. checks out the tag,
2. installs the Android platform and the build tools,
3. builds the release APK with the signing key from the secrets,
4. packs the web demo into a zip,
5. creates the GitHub release and attaches the APK and the zip.

The page of the release is then
`https://github.com/WanderingHumanid/mangokeys/releases/tag/v0.01`.

## One time setup: the signing key

The APK has to be signed with the same key every time, or Android will not
install an update over an older build. Generate a key once and keep it
somewhere safe:

```bash
keytool -genkeypair -v \
  -keystore mangokeys.keystore \
  -alias mangokeys \
  -keyalg RSA -keysize 2048 -validity 10000
```

The keystore is a secret. Do not commit it, the `.gitignore` already keeps
it out of the repository.

Add four secrets to the repository, under Settings, Secrets and variables,
Actions, New repository secret:

| Secret | What it is |
|---|---|
| `MANGOKEYS_KEYSTORE_BASE64` | the keystore, encoded: `base64 -w0 mangokeys.keystore` |
| `MANGOKEYS_KEYSTORE_PASSWORD` | the password of the keystore |
| `MANGOKEYS_KEY_ALIAS` | the alias, `mangokeys` in the command above |
| `MANGOKEYS_KEY_PASSWORD` | the password of the key |

**A release without a key is still possible.** When the secrets are missing
the workflow builds the APK and signs it with the debug key. It installs on
any phone, and Android will warn about the unknown source. That is enough for
a prototype, but a signed key is needed before there are real users, because
the debug key cannot update an installed app in place.

## One time setup: the model bundle

The build needs the model, the prefix model and the word list. They live in
`assets/release`, which is committed, so a build on a clean checkout finds
them. Rebuild the bundle after a new model is trained:

```bash
python -m scripts.build_release_assets
git add assets/release
git commit -m "Refresh the model bundle"
```

The manifest next to the files has the size and the checksum of each one.

## Publishing v0.01

```bash
cd /path/to/mangokeys

# 1. The tests have to pass.
python -m pytest tests -q
cd runtime && ./gradlew test && cd ..

# 2. The version in the app comes from the tag, but the commit needs to be
#    on the main branch first.
git checkout main
git pull --ff-only
git status

# 3. Tag it. The name of the tag is the version of the app.
git tag -a v0.01 -m "MangoKeys v0.01, the first public prototype"

# 4. Push the tag. This is what starts the release.
git push origin v0.01
```

The workflow builds the APK and creates the release page. Watch it here:
<https://github.com/WanderingHumanid/mangokeys/actions>

When the run is green the page is live and the APK can be downloaded from

```
https://github.com/WanderingHumanid/mangokeys/releases/download/v0.01/mangokeys-v0.01.apk
```

A link like that can be put in a README, in a message or anywhere else, and
it keeps working for every later version as well.

## If the workflow fails

The APK can also be built by hand, with the same commands the workflow runs:

```bash
cd runtime
export ANDROID_HOME=$HOME/Android/Sdk
./gradlew :app:assembleRelease
cd ..
# the APK is in apps/android/build/outputs/apk/release/app-release.apk
```

To create the release page by hand, use the GitHub web interface: go to
Releases, Draft a new release, choose the tag, and upload the APK. Or use
the `gh` command line tool:

```bash
gh release create v0.01 \
  --title "MangoKeys v0.01" \
  --notes-file docs/RELEASE_NOTES.md \
  apps/android/build/outputs/apk/release/app-release.apk
```

## The web demo

The Pages workflow publishes the web demo on every push to the main branch,
at <https://wanderinghumanid.github.io/mangokeys>. To build it by hand:

```bash
npm ci
python -m scripts.build_web --out dist/web
python apps/expo-web/serve.py --port 8000
```
