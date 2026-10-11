# Renders assets/generated/hls-estimate-cli.gif. Run from the repo root with the
# package installed in .venv: vhs assets/tapes/cli.tape
Output assets/generated/hls-estimate-cli.gif

Set Shell "bash"
Set FontSize 15
Set Width 1000
Set Height 480
Set Padding 24
Set Theme { "name": "hls-estimate", "background": "#131519", "foreground": "#e9e7e2", "cursor": "#e9e7e2", "selection": "#3e4550", "black": "#131519", "red": "#ff7a4d", "green": "#4fd1a5", "yellow": "#e0ad48", "blue": "#6f9cc0", "magenta": "#b98ad8", "cyan": "#9cc7e0", "white": "#e9e7e2", "brightBlack": "#868c96", "brightRed": "#ff9a75", "brightGreen": "#76e0bc", "brightYellow": "#eac170", "brightBlue": "#93b6d2", "brightMagenta": "#cda8e3", "brightCyan": "#b8d8ea", "brightWhite": "#ffffff" }
Set TypingSpeed 40ms

Hide
Type "export PATH=$PWD/.venv/bin:$PATH PS1='$ ' && clear"
Enter
Show

Type "python -m hls_estimate.cli estimate cnn_small --device zynq-7020"
Sleep 400ms
Enter
Sleep 6s
