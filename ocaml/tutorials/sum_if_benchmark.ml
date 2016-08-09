open Core.Std
open Core_bench.Std

let rec sum l =
  match l with
  | [] -> 0
  | hd :: tl -> hd + sum tl


let rec sum_if l =
	if List.is_empty l then 0
	else List.hd_exn l + sum_if (List.tl_exn l)

let display_config = Bench.Display_config.create
                       ~display:Textutils.Ascii_table.Display.column_titles
                       ~ascii_table:true
                       ~show_percentage:true
                       ()

let run_bench tests =
  Bench.bench
  ~display_config:display_config
  tests

let () =
	let numbers = List.range 0 1000 in
  [ Bench.Test.create ~name:"plus_one_match" (fun () ->
    ignore (sum numbers))
	; Bench.Test.create ~name:"plus_one_if" (fun () ->
	    ignore (sum_if numbers)) ]
	|> run_bench
