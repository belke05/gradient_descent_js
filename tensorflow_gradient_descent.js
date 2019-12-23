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
let m;
let b;
const LR = 0.5;
const EPOCHS = 10;
const optimizer = tf.train.sgd(LR);

setup();
function setup() {
  m = tf.variable(tf.scalar(Math.random()));
  b = tf.variable(tf.scalar(Math.random()));
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
  const y = Math.abs(b.dataSync()[0] - 1) * HEIGHT;
  const x = Math.abs(m.dataSync()[0] - 1) * WIDTH - y;
  console.log(x, y);
  m.print();
  b.print();
  ctx.clearRect(0, 0, WIDTH, HEIGHT);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  drawCircles();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "white";
  ctx.moveTo(0, y);
  ctx.lineTo(WIDTH, x);
  ctx.stroke();
}

function predict(m, b, tfxs) {
  return tfxs.mul(m).add(b);
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
        return loss(predict(m, b, tfxs), tfys);
      });
    }
  });
  // mse.innerText = `MSE: ${MSE.dataSync()[0].toString()}`;
  slope.innerText = `slope: ${m.dataSync()[0].toString()}`;
  intercept.innerText = `intercept: ${b.dataSync()[0].toString()}`;
  drawLine();
  //   for (let i = 0; i < EPOCHS; i++) {
  //     predictions.print();
  //     const MSE = error(tfys, predictions);
  //     MSE.print();
  //     if (MSE.dataSync()[0] > 2) break;
  //     const Dm = derivativeM(tfxs, tfys, predictions);
  //     const Dc = derivativeB(tfys, predictions);
  //     m = m.sub(LR * Dm.dataSync()[0]);
  //     b = b.sub(LR * Dc.dataSync()[0]);
  //
  //   }
  //   drawLine();
  console.log("changed m and b");
}
