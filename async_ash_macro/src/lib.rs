extern crate proc_macro;
mod commands;
mod commands_join;
mod transformer;

#[proc_macro]
pub fn commands(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    commands::proc_macro_commands(input.into()).into()
}

#[proc_macro]
pub fn join(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    commands_join::proc_macro_join(input.into()).into()
}
