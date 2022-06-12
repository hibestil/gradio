import gradio as gr

demo = gr.Blocks(css="#btn {color: red}")

with demo:
    default_json = {"a": "a"}

    num = gr.Variable(value=0)
    squared = gr.Number(value=0)
    btn = gr.Button("Next Square", elem_id="btn").style(rounded=False)

    stats = gr.Variable(value=default_json)
    table = gr.JSON()

    def increase(var, stats_history):
        var += 1
        stats_history[str(var)] = var**2
        return var, var**2, stats_history, stats_history

    btn.click(increase, [num, stats], [num, squared, stats, table])

if __name__ == "__main__":
    demo.launch()
