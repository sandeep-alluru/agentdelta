# redline demo — generate with: vhs assets/demo.tape
# Install vhs: https://github.com/charmbracelet/vhs

Output assets/demo.gif
Set FontSize 14
Set Width 900
Set Height 520
Set Theme "Dracula"
Set Padding 20
Set PlaybackSpeed 1.0

# Show the install
Type "pip install redline" Sleep 500ms Enter
Sleep 2s

# Clear and set scene
Ctrl+L

# Inspect the baseline trace
Type "redline inspect examples/baseline.jsonl" Sleep 300ms Enter
Sleep 2s

# Run the diff — regression case
Type "redline diff examples/baseline.jsonl examples/candidate.jsonl" Sleep 300ms Enter
Sleep 4s

# Show the fork point highlighted
Sleep 2s

# Exit code demo
Type "redline diff examples/baseline.jsonl examples/candidate.jsonl --exit-code; echo exit=$?" Sleep 300ms Enter
Sleep 2s
