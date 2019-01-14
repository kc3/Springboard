
function draw_circle(ctx, word, x, y, r) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.stroke();
}

function draw_text(ctx, text, x, y) {
    ctx.font = "30px Arial";
    ctx.fillText(text, x, y);
}

function draw_tree(tree_json) {
    var canvas = document.getElementById("treeCanvas");
    var ctx = canvas.getContext("2d");
    var width = 600;
    var height = 600;

    if(tree_json == null) {
        draw_text("No tree found!", 10, 50);
        return;
    };

    var obj = JSON.parse(tree_json)

    console.log(tree_json);
    ctx.fillText(obj["label"], 10, 50);
}