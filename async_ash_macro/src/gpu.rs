use crate::transformer::CommandsTransformer;

struct State {
}
impl Default for State {
    fn default() -> Self {
        Self{}
    }
}

impl CommandsTransformer for State {
    fn import(
        &mut self,
        input_tokens: &proc_macro2::TokenStream,
        is_image: bool,
    ) -> proc_macro2::TokenStream {
        todo!()
    }

    fn async_transform(&mut self, input: &syn::ExprAwait) -> syn::Expr {
        let base = input.base.clone();
        syn::Expr::Verbatim(quote::quote!{{
            let mut fut_pinned = std::pin::pin!(#base);
            fut_pinned.as_mut().init(__current_queue);
            loop {
                use ::async_ash::queue::QueueFuturePoll::*;
                match fut_pinned.as_mut().record(unsafe{&mut *(__ctx as *mut ::async_ash::queue::QueueContext)}) {
                    Ready { next_queue, output } => {
                        __current_queue = next_queue;
                        break output;
                    },
                    Semaphore => {
                        yield true;
                    },
                    Barrier => {
                        yield false;
                    }
                };
            }
        }})
    }
    //               queue
    // Leaf  nodes   the actual queue
    // Join  nodes   None, or the actual queue if all the same.
    // Block nodes   None
    // for blocks, who yields initially?
    // inner block yields. outer block needs to give inner block the current queue, and the inner block choose to yield or not.

    fn macro_transform(&mut self, mac: &syn::ExprMacro) -> syn::Expr {
        syn::Expr::Macro(mac.clone())
    }

    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr> {
        let returned_item = ret.expr.as_ref().map(|a| *a.clone()).unwrap_or(syn::Expr::Verbatim(quote::quote!(())));
        let token_stream = quote::quote!(
            {
                return (__current_queue, #returned_item);
            }
        );
        Some(syn::Expr::Verbatim(token_stream))
    }
}

pub fn proc_macro_gpu(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<crate::ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let mut state = State::default();

    let mut inner_closure_stmts: Vec<_> = input
        .stmts
        .iter()
        .map(|stmt| state.transform_stmt(stmt))
        .collect();
    if let Some(last) = inner_closure_stmts.last_mut() {
        if let syn::Stmt::Expr(expr) = last {
            let token_stream = quote::quote!({
                return (__current_queue, #expr);
            });
            *expr = syn::Expr::Verbatim(token_stream);
        } else {
            let token_stream = quote::quote!({
                return (__current_queue, ());
            });
            inner_closure_stmts.push(syn::Stmt::Expr(syn::Expr::Verbatim(token_stream)))
        }
    }
    quote::quote! {
        async_ash::queue::QueueFutureBlock::new(static |(__initial_queue, __ctx)| {
            let mut __current_queue: ::async_ash::queue::QueueRef = __initial_queue;
            #(#inner_closure_stmts)*
        })
    }
}
