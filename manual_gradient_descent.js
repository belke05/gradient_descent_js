const slope = document.getElementById("m");
const intercept = document.getElementById("b");
const mse = document.getElementById("mse");
const chart_container = document.getElementById("container");
const ctx = chart_container.getContext("2d");
ctx.strokeStyle = "red";
chart_container.addEventListener("mousedown", handleClick);
let xs = [];
let ys = [];
let xs_circle = [];
let ys_circle = [];
let m;
let b;
const LR = 0.1;
const EPOCHS = 10;

setup();

function handleClick(e) {
  const x = e.pageX - chart_container.offsetLeft;
  const y = e.pageY - chart_container.offsetTop;
  xs_circle.push(x);
  ys_circle.push(y);
  drawCircles();
  const Y = Math.abs(y - 400) / 400;
  const X = x / 400;
  console.log("new values", X, Y);
  xs.push(X);
  ys.push(Y);
  train(xs, ys);
}

function drawCircles() {
  ys_circle.forEach((y, i) => {
    ctx.strokeStyle = "red";
    console.log(y, xs_circle[i], "circles");
    ctx.moveTo(xs_circle[i], y);
    ctx.beginPath();
    ctx.arc(xs_circle[i], y, 2, 0, 2 * Math.PI);
    ctx.stroke();
  });
}

function drawLine() {
  const y = Math.abs(b.dataSync()[0] - 1) * 400;
  const x = m.dataSync()[0] * 400 + y;
  ctx.clearRect(0, 0, 400, 400);
  drawCircles();
  ctx.strokeStyle = "blue";
  ctx.moveTo(0, y);
  ctx.lineTo(400, x);
  ctx.stroke();
}

function setup() {
  m = tf.variable(tf.scalar(Math.random()));
  b = tf.variable(tf.scalar(Math.random()));
}

function predict(m, b, tfxs) {
  return tfxs.mul(m).add(b);
}

function error(tfys, predictions) {
  const error_sub = tf.sub(tfys, predictions);
  error_sub.print();
  const error_squared = tf.square(error_sub);
  error_squared.print();
  const MSE = error_squared.mean();
  return MSE;
}

function train(xs, ys) {
  const tfxs = tf.tensor1d(xs);
  const tfys = tf.tensor1d(ys);
  for (let i = 0; i < EPOCHS; i++) {
    const predictions = predict(m, b, tfxs);
    predictions.print();
    const MSE = error(tfys, predictions);
    MSE.print();
    if (MSE.dataSync()[0] > 2) break;
    const Dm = derivativeM(tfxs, tfys, predictions);
    const Dc = derivativeB(tfys, predictions);
    m = m.sub(LR * Dm.dataSync()[0]);
    b = b.sub(LR * Dc.dataSync()[0]);
    mse.innerText = `MSE: ${MSE.dataSync()[0].toString()}`;
    slope.innerText = `slope: ${m.dataSync()[0].toString()}`;
    intercept.innerText = `intercept: ${b.dataSync()[0].toString()}`;
  }
  drawLine();
  console.log("changed m and b");
}

function derivativeM(xs, ys, predictions) {
  const Dm = tf
    .sub(ys, predictions)
    .mul(xs)
    .sum()
    .mean()
    .mul(-2);
  Dm.print();
  return Dm;
}

function derivativeB(ys, predictions) {
  const Db = tf
    .sub(ys, predictions)
    .sum()
    .mean()
    .mul(-2);
  Db.print();
  return Db;
}
