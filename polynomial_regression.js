const slope = document.getElementById("m");
const intercept = document.getElementById("b");
const mse = document.getElementById("mse");
const chart_container = document.getElementById("container");
const WIDTH = chart_container.width;
const HEIGHT = chart_container.height;
const ctx = chart_container.getContext("2d");
ctx.strokeStyle = "red";
ctx.fillStyle = "black";
ctx.fillRect(0, 0, WIDTH, HEIGHT);
chart_container.addEventListener("mousedown", handleClick);
let xs = [];
let ys = [];
let xs_circle = [];
let ys_circle = [];
let a, b, c;
const LR = 0.1;
const EPOCHS = 10;
const optimizer = tf.train.sgd(LR);

setup();
function setup() {
  a = tf.variable(tf.scalar(Math.random()));
  b = tf.variable(tf.scalar(Math.random()));
  c = tf.variable(tf.scalar(Math.random()));
}

function handleClick(e) {
  const x = e.pageX - chart_container.offsetLeft;
  const y = e.pageY - chart_container.offsetTop;
  xs_circle.push(x);
  ys_circle.push(y);
  drawCircles();
  const Y = Math.abs(y - HEIGHT) / HEIGHT;
  const X = x / WIDTH;
  console.log("new values", X, Y);
  xs.push(X);
  ys.push(Y);
  train();
}

function drawCircles() {
  ys_circle.forEach((y, i) => {
    ctx.strokeStyle = "white";
    console.log(y, xs_circle[i], "circles");
    ctx.beginPath();
    ctx.arc(xs_circle[i], y, 2, 0, 2 * Math.PI);
    ctx.fillStyle = "white";
    ctx.fill();
    ctx.stroke();
  });
}

function drawLine() {
  const predictions = predict(a, b, c, range);
  predictions.print();
  const ranges = range.dataSync();
  ctx.clearRect(0, 0, WIDTH, HEIGHT);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  drawCircles();
  predictions.dataSync().forEach((pred, i) => {
    ctx.strokeStyle = "white";
    ctx.beginPath();
    // console.log(pred, "prediction");
    let y_cor = -1 * pred * HEIGHT;
    let x_cor = pred * HEIGHT;
    ctx.arc(ranges[i] * 400, (pred - 1) * -400, 2, 0, 2 * Math.PI);
    ctx.fillStyle = "white";
    ctx.fill();
    ctx.stroke();
  });
}

function predict(a, b, c, tfxs) {
  return tfxs
    .square()
    .mul(a)
    .add(tfxs.mul(b))
    .add(c);
}

function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}

function train() {
  const tfxs = tf.tensor1d(xs);
  const tfys = tf.tensor1d(ys);
  // will change the variables stored in tf
  tf.tidy(() => {
    if (xs.length > 0) {
      optimizer.minimize(() => {
        return loss(predict(a, b, c, tfxs), tfys);
      });
    }
  });
  // mse.innerText = `MSE: ${MSE.dataSync()[0].toString()}`;
  //   slope.innerText = `slope: ${m.dataSync()[0].toString()}`;
  //   intercept.innerText = `intercept: ${b.dataSync()[0].toString()}`;
  drawLine();
  a.print();
  b.print();
  c.print();
  console.log("changed a, b, c");
}

var ctxt = document.getElementById("myChart").getContext("2d");
function drawChart() {
  const range = tf.range(-1, 1, 0.001);
  var scatterChart = new Chart(ctxt, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Scatter Dataset",
          data: [
            {
              x: -10,
              y: 0
            },
            {
              x: 0,
              y: 10
            },
            {
              x: 10,
              y: 5
            }
          ]
        }
      ]
    },
    options: {
      scales: {
        xAxes: [
          {
            type: "linear",
            position: "bottom"
          }
        ]
      }
    }
  });
}
drawChart();
