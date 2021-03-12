$(document).ready(function() {

    const FUTURE_OPS = new Set(["X", "F", "U", "G", "WX", "R"]);
    const PAST_OPS = new Set(["Y", "O", "S", "H"]);

    $("#inputFormula").on("keyup", function () {
    $(".spinner-border").hide();
    var typed_formula = $("#inputFormula").val().split("");
    var upper_case = [];
    for (var i=0; i<typed_formula.length; i++){
      if (typed_formula[i] == typed_formula[i].toUpperCase())
        upper_case.push(typed_formula[i]);
    }
    var found_future = false;
    var found_past = false;
    for (var i=0; i<upper_case.length; i++){
      if (FUTURE_OPS.has(upper_case[i]))
        found_future = true;
      if (PAST_OPS.has(upper_case[i]))
        found_past = true;
    }
    if (found_future && found_past){
      $("#buttonFormula").attr("disabled", true);
      $("#Formula-alert").html("Error: You're typing a formula with both past and future operators.");
      $("#Formula-alert").show();

    }
    else{
      $("#Formula-alert").hide();
      $("#buttonFormula").attr("disabled", false);
    }
    });

    const example_formulas = [
      "(a & X(F((b & X(F(c))))))",
      "G((a -> F(b)))",
      "((F(a) & F(b)) & F(c))",
      "G(F((a & b)))",
      "F((a -> F((b & X(c)))))",
      "((toss & head) | ((toss & !(head)) & X(turn)))",
      "(((l & ta) & X (!oa)) | ((l & !(ta)) & X(oa)))"
    ];

    const controllables = [
      "a c",
      "a",
      "a c",
      "a",
      "a c",
      "toss turn",
      "l oa"
    ];

    const uncontrollables = [
      "b",
      "b",
      "b",
      "b",
      "b",
      "head",
      "ta"
    ];

    $("#rand-example").click(function () {
      // random select a formula from examples_formulas array
      var ran = Math.floor(Math.random() * example_formulas.length);
      $("#inputFormula").val(example_formulas[ran]);
      $("#inputFormula").focus();
      $("#controllables").val(controllables[ran]);
      $("#controllables").focus();
      $("#uncontrollables").val(uncontrollables[ran]);
      $("#uncontrollables").focus();
    });

});
