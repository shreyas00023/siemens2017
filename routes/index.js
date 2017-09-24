var express = require('express');
var router = express.Router();
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var expressValidator = require('express-validator');
var spawn = require('child_process').spawn;
router.use(bodyParser.urlencoded({ extended: true }));
/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Cervical Cancer Risk Assessment',heading: 'Cervical Cancer Risk Assessment',result:"Result Will Display Here"});
});
router.post('/run', function (req, res) {
    var process = spawn("python3",["CervicalCancerDT.py",req.body.age,req.body.partners,req.body.firstsex,req.body.numpreg,req.body.smokes,req.body.smokesyear,req.body.smokespack,req.body.contra,req.body.contrayears,req.body.iud,req.body.iudyears,req.body.std,req.body.stdnum,req.body.cond,req.body.cervc,req.body.vagc,req.body.vpc,req.body.syp,req.body.pid,req.body.gherp,req.body.mcont,req.body.aids,req.body.hiv,req.body.hpb,req.body.hpv,req.body.diag,req.body.diagtime,req.body.diagtimelast,req.body.dxcancer,req.body.dxcin,req.body.dxhpv]);
    process.stdout.on('data',function (data) {
        res.render('index', {title: 'Cervical Cancer Risk Assessment',heading: 'Cervical Cancer Risk Assessment',result:data});
    });
});

module.exports = router;
