{
  "targets": [
    {
      "target_name": "nlopt",
      "sources": [ "nlopt.cc" ],
       "include_dirs": [
	     "./nlopt-2.10.0/src/api/",
        "<!(node -e \"require('nan')\")"
	   ],
	    "dependencies": [
       		"./nlopt-2.10.0/nlopt.gyp:nloptlib"
    	]
    }
  ]
}
