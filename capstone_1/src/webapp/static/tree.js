
function get_color_value(label) {
    if (label == 0) {
        return "#FF0000";
    } else if (label == 1) {
        return "#ff9999";
    } else if (label == 2) {
        return "#FFFFFF";
    } else if (label == 3) {
        return "#3399ff";
    } else if (label == 4) {
        return "#0033cc";
    }

    return "#000000";
}

function get_circle_text(label) {
    if (label == 0) {
        return "--";
    } else if (label == 1) {
        return "-";
    } else if (label == 2) {
        return "0";
    } else if (label == 3) {
        return "+";
    } else if (label == 4) {
        return "++";
    }

    return "?";
}

function draw_circle(ctx, color, x, y, r) {
    console.log("draw_circle: " + x + " " + y);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fill();
}

function draw_line(ctx, x1, y1, x2, y2) {
    console.log("draw_line: " + x1 + " " + y1 + " " + x2 + " " + y2);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function draw_text(ctx, text, x, y) {
    console.log(x + " " + y);
    ctx.font = "15px Arial";
    ctx.strokeStyle = "#000000"
    ctx.strokeText(text, x, y);
}

function draw_node(ctx, node, x, y, hmargin, vmargin) {
    console.log("draw_node: " + x + " " + y);
    var r = 10;

    if (node["word"] == null) {
        if (node["left"]["word"] == null) {
            draw_line(ctx, x + node["left"]["width"] * hmargin, y, x + node["left"]["left"]["width"] * hmargin, y + vmargin);
        } else {
            draw_line(ctx, x + node["left"]["width"] * hmargin, y, x + hmargin - 10, y + vmargin);
        }

        if (node["right"]["word"] == null) {
            draw_line(ctx, x + node["left"]["width"] * hmargin, y, x + node["left"]["width"] * hmargin +
                node["right"]["left"]["width"] * hmargin, y + vmargin);
        } else {
            draw_line(ctx, x + node["left"]["width"] * hmargin, y,
                x + node["left"]["width"] * hmargin + hmargin - 10, y + vmargin);
        }

        draw_node(ctx, node["left"], x, y + vmargin, hmargin, vmargin);
        draw_node(ctx, node["right"], x + node["left"]["width"] * hmargin, y + vmargin, hmargin, vmargin);
        draw_circle(ctx, get_color_value(node["label"]), x + node["left"]["width"] * hmargin, y, r);
        draw_text(ctx, get_circle_text(node["label"]), x + node["left"]["width"] * hmargin - 5, y + 5)
    } else {
        draw_text(ctx, node["word"], x + hmargin - node["word"].length * 5, y + 10);
    }
}

function add_width_height(node) {
    if (node["word"] == null) {
        add_width_height(node["left"]);
        add_width_height(node["right"]);
        node["width"] = node["left"]["width"] + node["right"]["width"] + 1;
        node["height"] = Math.max(node["left"]["height"], node["right"]["height"]) + 1;
    } else {
        node["width"] = 1;
        node["height"] = 1;
    }
}

function draw_tree(tree_json, width, height) {
    var canvas = document.getElementById("treeCanvas");
    var ctx = canvas.getContext("2d");

    if(tree_json == null) {
        draw_text("No tree found!", 10, 50);
        return;
    };

    var obj = JSON.parse(tree_json);
    add_width_height(obj);
    console.log(JSON.stringify(obj, null, 2));

    var hmargin = (width - 40) / obj["width"];
    var vmargin = (height - 40) / obj["height"];

    console.log("Horizontal margin" + hmargin);
    console.log("Vertical margin" + vmargin);

    draw_node(ctx, obj, 20, 20, hmargin, vmargin);
}