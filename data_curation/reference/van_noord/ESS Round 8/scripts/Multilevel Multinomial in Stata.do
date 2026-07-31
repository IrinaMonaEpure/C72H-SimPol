*** Multilevel Multinomial with Loyalty as reference category
gsem (i.vote_behavior <- b2.class_community i.education hhincome female age religious urbanization ethnic_minority M1[country])
est store sem_results

* Margins and save to file 
est restore sem_results
margins, at(class_community=(1 2)) atmeans post
est store sem_margins

putexcel set "tables\Stata margins_expanded", modify
putexcel A1 = matrix(r(table)), names nfor(#.000)

* Mlincom - calculate differences/significane
est restore sem_margins
mlincom, clear
qui mlincom 2-1, 	add stat(est se ll ul p) rowname("Loyalty")
qui mlincom 4-3, 	add stat(est se ll ul p) rowname("Far left")
qui mlincom 6-5, 	add stat(est se ll ul p) rowname("Populist far left")
qui mlincom 8-7, 	add stat(est se ll ul p) rowname("Populist")
qui mlincom 10-9, 	add stat(est se ll ul p) rowname("Populist far right")
qui mlincom 12-11, 	add stat(est se ll ul p) rowname("Far right")
qui mlincom 14-13, 	add stat(est se ll ul p) rowname("Exit")
mlincom, twidth(30)


* Print regression results to rtf/word file
esttab sem_results ///
	using "stata_table2", rtf l interact( * ) one nogap noomit se(3) b(3) o(_cons) replace ///
	starl(+ 0.10 * 0.05 ** 0.01 *** 0.001) s(N) ///
	title("Table 6. Results from multilevel logistic regressions with voting behavior contrasts as dependent variables")	 